import torch
import torch.utils.data
import numpy as np
import itertools
import os
import random
import easydict
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

# Reference: torchvision `_box_cxcywh_to_xyxy`
def cxcywh_to_xyxy(boxes, clip=False):
    """
    Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    (cx, cy) refers to center of bounding box
    (w, h) are width and height of bounding box
    Args:
        boxes (Array[N, 4]): boxes in (cx, cy, w, h) format which will be converted.
        clip (bool): whether to clip out-of-bound values.

    Returns:
        boxes (Array(N, 4)): boxes in (x1, y1, x2, y2) format.
    """
    # We need to change all 4 of them so some temporary variable is needed.
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    
    boxes = np.stack((x1, y1, x2, y2), axis=-1)

    if clip:
        boxes = np.clip(boxes, 0., 1.)

    return boxes

class SAMDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path, train_shards, prob_use_caption, prob_use_boxes, box_confidence_th, batch_size, transform, *, max_boxes_per_image=32, shard_shuffle_seed=None, ddp_rank, num_ddp_processes, no_caption_only=False, return_cxcywh=False):
        if shard_shuffle_seed is not None:
            self.train_shards = np.copy(train_shards)
            
            shard_shuffle_rng = np.random.default_rng(seed=self.shard_shuffle_seed)
            shard_shuffle_rng.shuffle(train_shards)
        else:
            self.train_shards = train_shards
        
        self.transform = transform
        self.data_path = data_path
        self.shard_shuffle_seed = shard_shuffle_seed
        self.batch_size = batch_size
        self.prob_use_caption = prob_use_caption
        self.prob_use_boxes = prob_use_boxes
        self.box_confidence_th = box_confidence_th
        self.max_boxes_per_image = max_boxes_per_image
        self.return_cxcywh = return_cxcywh

        self.ddp_rank = ddp_rank
        self.num_ddp_processes = num_ddp_processes
        self.no_caption_only = no_caption_only

        print(f"data path: {self.data_path}, dataset init: rank: {self.ddp_rank}, num_ddp_processes: {self.num_ddp_processes}, no_caption_only: {no_caption_only}, return_cxcywh: {return_cxcywh}")
   
    def shard_iter(self, shard):
        latents = np.load(os.path.join(self.data_path, "latents", shard + ".npy"), allow_pickle=True).item()
        boxes = np.load(os.path.join(self.data_path, "boxes", shard + ".npy"), allow_pickle=True)
        
        # shuffle the saved boxes
        np.random.shuffle(boxes)
        
        latent_image_indices = latents["indices"]
        latent_arr = latents["latents"]

        image_to_latent_map = {}
        for latent_index, image_index in enumerate(latents["indices"]):
            image_to_latent_map[image_index] = latent_index
        
        for image_index, caption, boxes_raw, boxes_confidence, box_phrases in boxes:
            if image_index not in latent_image_indices:
                continue
            
            if random.uniform(0, 1) >= self.prob_use_boxes:
                # For `prob_use_boxes`, we use boxes. Otherwise, we ignore boxes.
                # Reference: https://github.com/gligen/GLIGEN/blob/f0ede1e5dc9e5f710fd564da297a3c1ba71a20b0/ldm/modules/diffusionmodules/openaimodel.py#L428
                th = 1.0
                if self.no_caption_only:
                    # also drop the caption if the boxes are dropped
                    # This is consistent with GLIGEN implementation: https://github.com/gligen/GLIGEN/blob/f9dccb9c6cf48bad03c3666290a7dec8c5e58f3c/demo/gligen/ldm/modules/diffusionmodules/openaimodel.py#L399
                    caption = ""
            else:
                if boxes_confidence.shape[0] > self.max_boxes_per_image:
                    th = np.max((self.box_confidence_th, np.sort(boxes_confidence)[-self.max_boxes_per_image]))
                else:
                    th = self.box_confidence_th
            
            if not self.return_cxcywh:
                boxes_raw = cxcywh_to_xyxy(boxes_raw, clip=True)
            boxes_mask = boxes_confidence > th
            boxes_raw = boxes_raw[boxes_mask]
            box_phrases = [box_phrase for box_phrase, box_mask in zip(box_phrases, boxes_mask) if box_mask]
            
            num_boxes = boxes_raw.shape[0]

            boxes_padded = np.zeros((self.max_boxes_per_image, 4))
            masks_padded = np.zeros(self.max_boxes_per_image)
            
            box_phrases = box_phrases[:self.max_boxes_per_image]
            boxes_padded[:num_boxes] = boxes_raw[:self.max_boxes_per_image]
            masks_padded[:num_boxes] = 1.
            
            if random.uniform(0, 1) >= self.prob_use_caption:
                # For `prob_use_caption`, we use caption. Otherwise, we ignore caption and only use boxes.
                caption = ""
            
            latents = latent_arr[image_to_latent_map[image_index]]
            # image_index is numpy number
            output = dict(id=int(image_index), caption=caption, boxes=torch.tensor(boxes_padded), box_phrases=box_phrases, text_masks=torch.tensor(masks_padded), latents=torch.tensor(latents))
        
            if self.transform:
                output = self.transform(output)
            
            yield output
   
    def __iter__(self, worker_info=None):
        if worker_info is None:
            worker_info = torch.utils.data.get_worker_info()
        # Every DDP process has workers
        worker_id = worker_info.id + self.ddp_rank * worker_info.num_workers
        num_workers = worker_info.num_workers * self.num_ddp_processes
        seed = worker_info.seed % 2**32
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Reference: https://stackoverflow.com/a/69779930
        while True:
            # This is for simplicity: a worker always loads a certain set of numpy files
            worker_shards = itertools.islice(self.train_shards, worker_id, None, num_workers)
            for shard in worker_shards:
                print("Loading shard", shard)
                shard_iter = self.shard_iter(shard)
                
                batch = []
                for iter_item in shard_iter:
                    batch.append(iter_item)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []

if __name__ == "__main__":
    # python dataset/sam_dataset.py 
    dataset = SAMDataset(data_path="data_256", train_shards=[f"sa_{i:06}" for i in range(400, 500, 1)], prob_use_caption=0.5, prob_use_boxes=0.9, box_confidence_th=0.25, batch_size=2, transform=None, shard_shuffle_seed=None, ddp_rank=0, num_ddp_processes=1)
    
    # Simulate worker 1 (not 0)
    dataset_iter = dataset.__iter__(worker_info=easydict.EasyDict(id=1, num_workers=2, seed=0))
    for idx, item in enumerate(dataset_iter):
        print(f"Index: {idx}")
        print(item)
        
        if idx >= 4:
            break
