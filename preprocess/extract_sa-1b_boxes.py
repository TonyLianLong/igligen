# Note: part of this file is from GroundingDINO.

import pandas as pd
import os
import tqdm
import glob
import pyarrow.parquet as pq
import bisect
import torch
from groundingdino.util.inference import load_model, preprocess_caption, get_phrases_from_posmap, annotate
import groundingdino.datasets.transforms as T
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List
from torchvision import transforms
import numpy as np
import tarfile
import io

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed = transform(image_source)
    return image, image_transformed


def stream_tar_contents(tar_file_path, no_transform=False):
    with tarfile.open(tar_file_path, "r:gz") as tar:
        for member in tar:
            if member.isfile():
                if os.path.splitext(member.name)[-1] == ".jpg":
                    file_obj = tar.extractfile(member)
                    if file_obj:

                        if no_transform:
                            img = np.asarray(Image.open(file_obj).convert("RGB"))
                        else:
                            _, img = load_image(file_obj)
                        
                        file_obj.close()
                        
                        yield (member.name, img)

# Threshold for obtaining boxes.
# box_threshold = 0.35
box_threshold = 0.2
text_threshold = 0.2

# This is the resolution for GroundingDINO. After getting the boxes we can use it with latents from images of different resolution without changing this resolution.
resolution = 512

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
                    caption = all_captions.iloc[keys.index(index)]["caption"]
                    caption = caption.replace("The image features ", "")
                    caption = preprocess_caption(caption=caption)

                    info = dict(image_path=image_path, index=index)

                    yield image, caption, info
                except ValueError as e:
                    print(f"Error: {e}, skipping file {filename} index {index}")
                    continue


def process_model_outputs(
    prediction_logits, 
    prediction_boxes, 
    captions,
    box_threshold: float,
    text_threshold: float,
    remove_combined: bool = False
):
    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(captions)
    
    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
        
        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]

    return boxes, logits.max(dim=1)[0], phrases


@torch.no_grad()
def predict(
        model,
        images: torch.Tensor,
        captions: List[str],
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    outputs = model(images, captions=captions)

    prediction_probs = outputs["pred_logits"].cpu().sigmoid()  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()  # prediction_boxes.shape = (nq, 4)

    boxes_all, probs_all, phrases_all = [], [], []
    for prediction_probs_item, prediction_boxes_item, captions_item in zip(prediction_probs, prediction_boxes, captions):
        boxes, probs, phrases = process_model_outputs(prediction_probs_item, prediction_boxes_item, captions_item, box_threshold=box_threshold, text_threshold=text_threshold, remove_combined=remove_combined)

        boxes_all.append(boxes)
        probs_all.append(probs)
        phrases_all.append(phrases)
    
    return boxes_all, probs_all, phrases_all


@torch.no_grad()
def extract_boxes(images, captions):
    boxes_all, probs_all, phrases_all = predict(
        model=model,
        images=images,
        captions=captions,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        remove_combined=True
    )
    
    return boxes_all, probs_all, phrases_all

def run(dataloader):
    box_info = []

    for ind, (images, captions, info) in enumerate(tqdm.tqdm(dataloader)):
        images = images.to("cuda", non_blocking=True)[None]
        captions = [captions]
        info['index'] = [info['index']]
        
        boxes_all, probs_all, phrases_all = extract_boxes(images, captions)
        
        for index, caption, boxes_item, probs_item, phrases_item in zip(info['index'], captions, boxes_all, probs_all, phrases_all):
            index, boxes_item, probs_item, phrases_item = index, boxes_item.numpy().astype(np.float16), probs_item.numpy().astype(np.float16), phrases_item
            box_info.append([index, caption, boxes_item, probs_item, phrases_item])

    
    box_info = sorted(box_info, key=lambda item: item[0])
    
    np.save(save_path, np.array(box_info, dtype=object))
    print(f"Saved to {save_path}")


import sys

tar_files = [sys.argv[1]]
print(tar_files)

dataset = Dataset(tar_files)
assert len(dataset.tar_files) == 1
filename = os.path.splitext(dataset.tar_files[0].split("/")[-1])[0] + ".npy"
os.makedirs("boxes", exist_ok=True)
save_path = "boxes/" + filename
if os.path.exists(save_path):
    print(f"File {save_path} exists, skipping")
    exit()

all_captions = pq.read_table(os.path.expanduser("path_to_SA1B_caption.parquet")).to_pandas()

keys = all_captions["key"].str.replace("sa_", "").astype(int).tolist()

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
model = model.to("cuda")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=1, pin_memory=True)
run(dataloader)
