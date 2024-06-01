# IGLIGEN: Improved Implementation for Training GLIGEN (Open-Set Grounded Text-to-Image Generation)
This research project aims to create a complete, huggingface-style pipeline for **training** GLIGEN adapters. The project is part of the effort in creating [LLM-grounded Diffusion Models (LMD+)](https://llm-grounded-diffusion.github.io/) and [LLM-grounded Video Diffusion Models (LVD-GLIGEN)](https://llm-grounded-video-diffusion.github.io/). **This implementation supports SD v1.4/v1.5, SD v2.0/v2.1, and ModelScope (grounded text-to-video generation), with SDXL support planned.** You can use/download pretrained GLIGEN adapters [here](#pretrained-gligen-adapters).

## Motivation
The official [GLIGEN](https://github.com/gligen/GLIGEN) training code is from the original latent diffusion/stable diffusion codebase. This make  it not fully compatible with huggingface `diffusers`, `transformers`, `datasets`, and `accelerate` that are commonly used for training diffusion models. Currently, the only released gligen weights are on SDv1.4, and weights for more updated models such as SDv2.1 are missing. This repo (IGLIGEN) makes training GLIGEN on custom datasets and custom models easier.

## TODO
- [ ] Add SDXL support
- [ ] Add models with further fine-tuning with [needle in a haystack](https://arxiv.org/abs/2309.15807), using aesthetic datasets such as [LAION Aesthetics](https://laion.ai/blog/laion-aesthetics/).

Please send me a message if you're interested in helping with the project.

## Examples
### IGLIGEN on SDv2.1 for box-controlled text-to-image generation
Condition for the examples:
Prompt: "An image of grassland with a dog and a tree."

GLIGEN phrases: `["a dog", "a tree"]`

GLIGEN boxes: `[[0.1, 0.6, 0.3, 0.8], [0.6, 0.2, 0.9, 0.8]]`

| Baseline (generated images at the beginning of training) | IGLIGEN (generated images at the end of training) | 
| ------ | -------- | 
| ![image](https://github.com/TonyLianLong/igligen/assets/1451234/887b21db-a0bf-49bb-9eda-354a806e9e02) | ![image](https://github.com/TonyLianLong/igligen/assets/1451234/ee54016a-0fd4-4811-a684-cba180c9407d) |

### IGLIGEN on Zeroscope for box-controlled text-to-video generation
Example videos generated with the scripts above with prompt `A bear walking from the left to the right`, when combined with our work [LLM-grounded Video Diffusion Models](https://llm-grounded-video-diffusion.github.io/):
| Baseline (Zeroscope) | LVD-GLIGEN (using IGLIGEN adapters) | 
| ---- | -------- | 
| ![Example Video Demo: Zeroscope baseline](https://github.com/TonyLianLong/LLM-groundedVideoDiffusion/blob/main/assets/example_video_zeroscope_baseline.gif) | ![Example Video Demo: LVD on Zeroscope](https://github.com/TonyLianLong/LLM-groundedVideoDiffusion/blob/main/assets/example_video_lvd_gligen_zeroscope.gif) |

Note that the gifs are compressed so the actual generation quality is higher.

## Pretrained GLIGEN adapters
### Text-to-video generation
The pretrained adapters for text-to-video generation on ModelScope can be found here: [https://huggingface.co/longlian/text-to-video-lvd-ms](https://huggingface.co/longlian/text-to-video-lvd-ms). The pretrained adapters for text-to-video generation on Zeroscope can be found here: [https://huggingface.co/longlian/text-to-video-lvd-zs](https://huggingface.co/longlian/text-to-video-lvd-zs). 

You can use LLM-generated dynamic scene layouts (i.e., stage 1 of LVD) or provide the boxes for each frame. Check out the [example colab](https://colab.research.google.com/drive/17He4bFAF8lXmT9Nfv-Sg29iKtPelDUNZ) with Modelscope (square videos, sometimes with watermark). Check out the [example colab](https://colab.research.google.com/drive/1ySd_Ja2SZFQ1UbHz7dAChDO_nERM_B6H) with Zeroscope (horizontal videos).

<details>
  <summary>An example that uses ModelScope + GLIGEN.</summary>

Tested with `diffusers==0.27.2`.

```python
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import imageio
from IPython.display import Image as IPyImage
import numpy as np

pipe = DiffusionPipeline.from_pretrained("longlian/text-to-video-lvd-ms", trust_remote_code=True, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "An image of grassland with a dog walking from the left to the right."
fps = 4
num_frames = 16
lvd_gligen_boxes = [
  [[0.15, 0.6, 0.5, 0.8]],
  [[0.19, 0.6, 0.54, 0.8]],
  [[0.22999999999999998, 0.6, 0.58, 0.8]],
  [[0.27, 0.6, 0.62, 0.8]],
  [[0.31, 0.6, 0.6599999999999999, 0.8]],
  [[0.35, 0.6, 0.7, 0.8]],
  [[0.39, 0.6, 0.74, 0.8]],
  [[0.43000000000000005, 0.6, 0.78, 0.8]],
  [[0.47, 0.6, 0.82, 0.8]],
  [[0.51, 0.6, 0.86, 0.8]],
  [[0.55, 0.6, 0.9, 0.8]],
  [[0.59, 0.6, 0.94, 0.8]],
  [[0.63, 0.6, 0.98, 0.8]],
  [[0.67, 0.6, 1.02, 0.8]],
  [[0.7100000000000001, 0.6, 1.06, 0.8]],
  [[0.75, 0.6, 1.1, 0.8]]
]
lvd_gligen_phrases = [["a dog"] for _ in range(num_frames)]

generator = torch.manual_seed(1)
video_frames = pipe(prompt, num_inference_steps=25, height=256, width=256, num_frames=16, lvd_gligen_scheduled_sampling_beta=1.0, lvd_gligen_boxes=lvd_gligen_boxes, lvd_gligen_phrases=lvd_gligen_phrases, generator=generator).frames
video = imageio.mimsave(imageio.RETURN_BYTES, video_frames, format='gif', loop=0, duration=1000 * 1/fps)
display(IPyImage(data=video, format='gif'))
```
</details>

### Text-to-image generation
The pretrained adapters for text-to-video generation on SDv2.1 can be found here: [https://huggingface.co/longlian/igligen-sd2.1-v1.0](https://huggingface.co/longlian/igligen-sd2.1-v1.0). The models trained with this repo is compatible with the official pipeline for inference (`StableDiffusionGLIGENPipeline`) on SDv1.5 and SDv2.1. You can use it in LMD+, as currently GLIGEN only offers weights for SDv1.4. Check out the [example colab](https://colab.research.google.com/drive/1vl3Y2gZcjmXBh7fdDUMF9v-rrQZBXUD7).

<details>
  <summary>An example that uses SD v2.1 + GLIGEN.</summary>

Tested with `diffusers==0.27.2`.

```python
import torch
from diffusers import StableDiffusionGLIGENPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionGLIGENPipeline.from_pretrained("longlian/igligen-sd2.1-v1.0", torch_dtype=torch.float16).to("cuda")
pipe = pipe.to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "An image of grassland with a dog."

images = pipe(prompt, num_inference_steps=25, height=512, width=512, gligen_scheduled_sampling_beta=0.4, gligen_boxes=[[0.1, 0.6, 0.3, 0.8]], gligen_phrases=["a dog"], num_images_per_prompt=1).images

for image in images:
    display(image)
```
</details>

## Dataset Download and Preprocessing
**Note: we have preprocessed the dataset and uploaded them to huggingface. See [Download Preprocessed Dataset](#download-preprocessed-dataset) for details. You only need to download the raw dataset if you want to preprocess them on your own.**

[GLIGEN](https://github.com/gligen/GLIGEN) uses a mix of multiple datasets in training, with text/image embeddings coded in the tsv files. The preprocessing code (how to turn a dataset into the tsv files) is not given, which makes it hard to adapt to custom datasets. The text embeddings are different for different diffusion models, which make the stored embeddings in the tsv files not useful. The dataset is thus not efficiently stored.

This project takes another route. Similar to PixArt-alpha, we use [SA-1B](https://ai.meta.com/datasets/segment-anything/) dataset for training. Note that SA-1B dataset actually has ~11M images. If you use the code/weights from this project, you agree to adhere to the [terms and conditions](https://ai.meta.com/datasets/segment-anything-downloads/) of SA-1B.

We provide complete preprocessing scripts, so you can easily switch to other datasets on your own.

We also have an implementation that takes in the tsv files of GLIGEN. However, it's not as efficient, and the dataloader part is more complicated. Therefore, it's not added to this repo. If you would like to get the code for this variant, you can create an issue.

### Download Raw Dataset
**Note: See [Download Preprocessed Dataset](#download-preprocessed-dataset) for preprocessed dataset.**

Please download the tar files from the [SA-1B website](https://ai.meta.com/datasets/segment-anything-downloads/). Captions of images of SA-1B are from LLaVA, generated by PixArt-alpha team, you can download them here [SAM-LLaVA-captions](https://huggingface.co/datasets/PixArt-alpha/SAM-LLaVA-Captions10M).

### Dataset Preprocessing
We use [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) to obtain the boxes. The [script](preprocess/extract_sa-1b_boxes.py) is in `preprocessing` directory. Please update the paths to SAM-LLaVA-captions. You probably need to clone GroundingDINO and put the script inside. Please refer to the [GroundingDINO repo](https://github.com/IDEA-Research/GroundingDINO) for GroundingDINO config and weights.

We also preprocess the dataset into latents so that it does not have to be encoded again in training. This also saves a lot of space. Note that you need to preprocess the dataset for the desired resolution (e.g., SDv1.4/1.5/2.1-base use 512x512, while ModelScope uses 256x256). The [script](preprocess/encode_latents.py) is in `preprocessing` directory.

You can use `xargs` command to parallelize preprocessing:
```shell
ls -1 /path_to_sa-1b/sa_000{000..999}.tar|sort|xargs -P 2 -I {} python extract_sa-1b_boxes.py {}
ls -1 /path_to_sa-1b/sa_000{000..999}.tar|sort|xargs -P 2 -I {} python encode_latents.py {}
```

### Download Preprocessed Dataset
The dataset is preprocessed and stored in latent format. The preprocessed dataset is much smaller than the original dataset, thanks to the efficient latent space. As an example, SA-1B tar files are typically around 11GB. However, after preprocessing, each npy file corresponding to a tar file is around 368MB (resolution is 512x512). Only the preprocessed dataset is used in training.

| Preprocessed SA-1B | HuggingFace Link |
| -------- | ------- |
| Latents (512x512, for SD v1.4/1.5/2.1-base)  | https://huggingface.co/datasets/longlian/sa-1b_latents_512 |
| Latents (256x256, for ModelScope) | https://huggingface.co/datasets/longlian/sa-1b_latents_256 |
| Latents (768, for SD v2.1)  | https://huggingface.co/datasets/longlian/sa-1b_latents_768 |
| Bounding Boxes    | https://huggingface.co/datasets/longlian/sa-1b_boxes |

Please put the latents in `data/latents` and bounding boxes in `data/boxes`.

## Training
Training with 4 GPUs with SD v2.1:

```shell
sh train_sdv2.1.sh
```

You can edit the training script to use more/fewer GPUs and update the hyperparameters.

The training time for 500k steps is roughtly 3.75 days on 4x A6000 with image resolution 512x512 on SDv2.1.

## Citation
The authors of this repo (IGLIGEN) are not affiliated with the authors of GLIGEN. Since IGLIGEN is based on GLIGEN, if you use the IGLIGEN code or adapters, please kindly consider citing the original GLIGEN paper:
```
@article{li2023gligen,
  title={GLIGEN: Open-Set Grounded Text-to-Image Generation},
  author={Li, Yuheng and Liu, Haotian and Wu, Qingyang and Mu, Fangzhou and Yang, Jianwei and Gao, Jianfeng and Li, Chunyuan and Lee, Yong Jae},
  journal={CVPR},
  year={2023}
}
```

The project is part of the effort in creating [LLM-grounded Diffusion Models (LMD+)](https://llm-grounded-diffusion.github.io/) and [LLM-grounded Video Diffusion Models (LVD-GLIGEN)](https://llm-grounded-video-diffusion.github.io/). 

Please kindly consider citing LMD if you use IGLIGEN code/trained weights in image generation setting.
```
@article{lian2023llmgrounded,
  title={Llm-grounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models},
  author={Lian, Long and Li, Boyi and Yala, Adam and Darrell, Trevor},
  journal={arXiv preprint arXiv:2305.13655},
  year={2023}
}
```

Please kindly consider citing LVD if you use IGLIGEN code/trained weights in video generation setting.
```
@article{lian2023llmgroundedvideo,
  title={Llm-grounded video diffusion models},
  author={Lian, Long and Shi, Baifeng and Yala, Adam and Darrell, Trevor and Li, Boyi},
  journal={arXiv preprint arXiv:2309.17444},
  year={2023}
}
```
