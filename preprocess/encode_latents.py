# Note: part of this file is from GroundingDINO and HF SD pipeline.

from diffusers import AutoencoderKL

import os
import tqdm
import torch
from PIL import Image
from typing import Tuple
from torchvision import transforms
import numpy as np
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tarfile


def stream_tar_contents(tar_file_path, no_transform=False):
    # Open the tar file in streaming mode
    with tarfile.open(tar_file_path, "r:gz") as tar:
        for member in tar:
            # Check if the member is a regular file
            if member.isfile():
                if os.path.splitext(member.name)[-1] == ".jpg":
                    # print(member.name)
                    # Extract file object for the current member
                    file_obj = tar.extractfile(member)
                    if file_obj:
                        if no_transform:
                            img = np.asarray(Image.open(file_obj).convert("RGB"))
                        else:
                            _, img = load_image(file_obj)

                        file_obj.close()

                        yield (member.name, img)


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, tar_files):
        self.tar_files = tar_files

    def __iter__(self):
        for tar_file in self.tar_files:
            content = stream_tar_contents(tar_file)
            for image_path, image in content:
                filename = image_path.split("/")[-1]
                index = int(os.path.splitext(filename)[0].replace("sa_", ""))
                try:
                    info = dict(image_path=image_path, index=index)

                    yield image, info
                except ValueError as e:
                    print(f"Error: {e}, skipping file {filename} index {index}")
                    continue


# Note that SDv1.4/1.5/2.0/2.1/Modelscope use the same VAE encoder
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
revision = None
variant = None
# You need to change this resolution to store latents for different SD training resolution. For example, SDv1.4/1.5/2.1-base use the same resolution.
resolution = 512


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = transforms.Compose(
        [
            transforms.Resize(
                resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            # vae normalization coefficients
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed = transform(image_source)
    return image, image_transformed


@torch.no_grad()
def decode(vae, latents):
    sdxl = hasattr(vae.config, "scaling_factor")
    # make sure the VAE is in float32 mode, as it overflows in float16

    if sdxl:
        image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    else:
        scaled_latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(scaled_latents).sample

    image = image.float()

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")

    return images


@torch.no_grad()
def vae_encode(images):
    latents = vae.encode(images.to(torch.float32)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    return latents


tar_files = [sys.argv[1]]
print(tar_files)

dataset = Dataset(tar_files)
assert len(dataset.tar_files) == 1
os.makedirs("latents", exist_ok=True)
filename = os.path.splitext(dataset.tar_files[0].split("/")[-1])[0] + ".npy"
save_path = "latents/" + filename
if os.path.exists(save_path):
    print(f"File {save_path} exists, skipping")
    exit()

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=None, num_workers=1, pin_memory=True
)


vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant
).cuda()


index_latents = []

for ind, (images, info) in enumerate(tqdm.tqdm(dataloader)):
    images = images.to("cuda", non_blocking=True)[None]
    info["index"] = [info["index"]]

    latents = vae_encode(images)

    for index, latents_item in zip(info["index"], latents):
        index, latents_item = index, latents_item.to(torch.float16).cpu().numpy()
        index_latents.append((index, latents_item))


index_latents = sorted(index_latents, key=lambda item: item[0])


indices_all = [indices for indices, _ in index_latents]
latents_all = np.stack([latents for _, latents in index_latents], axis=0)

index_latents_all = dict(indices=indices_all, latents=latents_all)
np.save(save_path, np.array(index_latents_all, dtype=object))

print(f"Saved to {save_path}")
