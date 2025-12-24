import torch.nn.init as init
import argparse
from cgitb import text
import copy
import gc
import itertools
import logging
import math
import os
import random
import shutil
from tkinter import NO
import warnings
from contextlib import nullcontext
from pathlib import Path
import PIL.Image
import PIL.ImageOps
import numpy as np
from sympy import N
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast
import cv2
import json
from typing import List, Tuple

from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline,
    # SD3Transformer2DModel,
    # StableDiffusion3InstructPix2PixPipeline
)
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

import accelerate
import datasets
import PIL
import requests
import torch.nn as nn
import torch.nn.functional as F
from os.path import join
from datasets import load_dataset
from packaging import version



from transformer_sd3 import SD3Transformer2DModel
from pipeline_stable_diffusion_3_instructpix2pix_crop_blending import StableDiffusion3InstructPix2PixPipeline_blending


def crop_square_img(mask_img, bbox, crop_size=512, min_len=256, expand_rate=1.6):
    # ori mask size
    img_width, img_height = mask_img.size
    
    # Extract bounding box coordinates
    xmin, ymin, xmax, ymax = bbox
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin
    
    # Find the center of the bounding box
    center_x = xmin + bbox_width // 2
    center_y = ymin + bbox_height // 2
    
    # Determine the side length of the square, which should be the larger of bbox width or height
    side_length = max(bbox_width, bbox_height)
    
    # Adjust side_length to be at least 20% larger than the original rectangle's larger side
    side_length = int(side_length * expand_rate)
    side_length = max(side_length, min_len)
    side_length = min(side_length, max(img_width, img_height))
    side_length = min(side_length, crop_size)
    
    # Calculate new square coordinates
    new_xmin = center_x - side_length // 2
    new_ymin = center_y - side_length // 2
    new_xmax = new_xmin + side_length
    new_ymax = new_ymin + side_length
    
    # Create a new blank (black) image with the same mode and square size
    square_crop = Image.new(mask_img.mode, (side_length, side_length), color=0)
    
    # Calculate the region of the original image that overlaps with the new square
    overlap_xmin = max(new_xmin, 0)
    overlap_ymin = max(new_ymin, 0)
    overlap_xmax = min(new_xmax, img_width)
    overlap_ymax = min(new_ymax, img_height)
    
    # Calculate the position where this region should be pasted on the new square image
    paste_x = overlap_xmin - new_xmin
    paste_y = overlap_ymin - new_ymin
    
    # Crop the overlapping region from the original image
    cropped_mask_region = mask_img.crop((overlap_xmin, overlap_ymin, overlap_xmax, overlap_ymax))
    
    # Paste the cropped region onto the black square image
    square_crop.paste(cropped_mask_region, (paste_x, paste_y))
    # square_crop = square_crop.resize((crop_size, crop_size))
    
    return square_crop


def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask


def bbox_to_mask(bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int], expand_ratio: float = 0.1) -> np.ndarray:
    """
    Convert a bounding box to a segmentation mask with optional expansion.

    Args:
    - bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax).
    - image_shape (tuple): Shape of the image (height, width) for the mask.
    - expand_ratio (float): Ratio to expand the rectangle (default 0.1 for 10%).

    Returns:
    - np.ndarray: Segmentation mask with the expanded rectangle filled.
    """
    xmin, ymin, xmax, ymax = bbox
    height, width = image_shape

    w = xmax - xmin
    h = ymax - ymin
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    w_exp = w * (1 + expand_ratio)
    h_exp = h * (1 + expand_ratio)

    xmin_new = max(int(cx - w_exp / 2), 0)
    ymin_new = max(int(cy - h_exp / 2), 0)
    xmax_new = min(int(cx + w_exp / 2), width - 1)
    ymax_new = min(int(cy + h_exp / 2), height - 1)

    mask = np.zeros(image_shape, dtype=np.uint8)

    mask[ymin_new:ymax_new+1, xmin_new:xmax_new+1] = 255

    return mask



def crop_square_for_blending(mask_img, bbox, crop_size=512, min_len=192, expand_rate=1.2, core_expand_rate=1.1):
    # ori mask size
    img_width, img_height = mask_img.size
    
    # Extract bounding box coordinates
    xmin, ymin, xmax, ymax = bbox
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin
    
    # Find the center of the bounding box
    center_x = xmin + bbox_width // 2
    center_y = ymin + bbox_height // 2
    
    # Determine the side length of the square, which should be the larger of bbox width or height
    side_length = max(bbox_width, bbox_height)
    
    # Adjust side_length to be at least 20% larger than the original rectangle's larger side
    side_length = int(side_length * expand_rate)
    side_length = max(side_length, min_len)
    side_length = min(side_length, max(img_width, img_height))
    side_length = min(side_length, crop_size)

    
    # Calculate new square coordinates 
    new_xmin = center_x - side_length // 2
    new_ymin = center_y - side_length // 2
    new_xmax = new_xmin + side_length
    new_ymax = new_ymin + side_length
    
    # Create a new blank (black) image with the same mode and square size
    square_crop = Image.new(mask_img.mode, (side_length, side_length), color=0)
    
    # Calculate the region of the original image that overlaps with the new square
    overlap_xmin = max(new_xmin, 0)
    overlap_ymin = max(new_ymin, 0)
    overlap_xmax = min(new_xmax, img_width)
    overlap_ymax = min(new_ymax, img_height)
    
    # Calculate the position where this region should be pasted on the new square image
    paste_x = overlap_xmin - new_xmin
    paste_y = overlap_ymin - new_ymin
    
    # Crop the overlapping region from the original image
    cropped_mask_region = mask_img.crop((overlap_xmin, overlap_ymin, overlap_xmax, overlap_ymax))

    original_coor = (overlap_xmin, overlap_ymin, overlap_xmax, overlap_ymax)
    coor_in_crop = (paste_x, paste_y, paste_x+(overlap_xmax-overlap_xmin), paste_y+(overlap_ymax-overlap_ymin))
    
    # Paste the cropped region onto the black square image
    square_crop.paste(cropped_mask_region, (paste_x, paste_y))
    square_crop = square_crop.resize((crop_size, crop_size))


    # core region
    core_side_length = max(bbox_width, bbox_height)
    core_side_length = int(core_side_length * core_expand_rate)
    core_side_length = min(core_side_length, max(img_width, img_height))
    core_side_length = min(core_side_length, crop_size)

    core_new_xmin = center_x - core_side_length // 2
    core_new_ymin = center_y - core_side_length // 2
    core_new_xmax = core_new_xmin + core_side_length
    core_new_ymax = core_new_ymin + core_side_length

    core_overlap_xmin = max(core_new_xmin, 0)
    core_overlap_ymin = max(core_new_ymin, 0)
    core_overlap_xmax = min(core_new_xmax, img_width)
    core_overlap_ymax = min(core_new_ymax, img_height)

    core_paste_x = core_overlap_xmin - new_xmin
    core_paste_y = core_overlap_ymin - new_ymin

    square_blend = Image.new('L', (side_length, side_length), color=0)
    square_gen = Image.new('L', (core_overlap_xmax-core_overlap_xmin, core_overlap_ymax-core_overlap_ymin), color=255)
    square_blend.paste(square_gen, (core_paste_x, core_paste_y))
    square_blend = square_blend.resize((crop_size, crop_size), resample=Image.NEAREST)

    return square_crop, square_blend, original_coor, coor_in_crop, side_length



class IIE_Dataset(Dataset):
    def __init__(self, split='train', resolution=512, CenterCrop=False, Flip=True, ExpRate=1.2, MinCrop=192):
        self.split = split
        self.resolution = resolution
        self.ExpRate = ExpRate
        self.MinCrop = MinCrop

        seededit_object_dataset = ''
        ultraedit_object_dataset = ''



        ADD_NUM = 0
        ROMV_NUM = 0
        CHANGE_NUM = 0
        WHOLE_IMAGE = 0
        if self.split == 'train':
            self.train_data_list = []

            for ss in os.listdir(seededit_object_dataset):
                for tt in ['object_removal', 'object_addition']:
                    curr_tt_folder = os.path.join(seededit_object_dataset, ss, tt)
                    for dd in os.listdir(curr_tt_folder):
                        if tt == 'object_removal':
                            ROMV_NUM += 1
                        if tt == 'object_addition':
                            ADD_NUM += 1
                            self.train_data_list.append(os.path.join(seededit_object_dataset, ss, tt, dd))
                        self.train_data_list.append(os.path.join(seededit_object_dataset, ss, tt, dd))


            for ss in os.listdir(ultraedit_object_dataset):
                for tt in ['object_change', 'object_addition']:
                    curr_tt_folder = os.path.join(ultraedit_object_dataset, ss, tt)
                    if not os.path.isdir(curr_tt_folder):
                        continue
                    for dd in os.listdir(curr_tt_folder):
                        full_path = os.path.join(curr_tt_folder, dd)
                        if not os.path.isdir(full_path):
                            continue
                        if tt == 'object_change':
                            CHANGE_NUM += 1
                            self.train_data_list.append(full_path)
                        elif tt == 'object_addition':
                            ADD_NUM += 1
                            self.train_data_list.append(full_path)

            self.train_transforms = transforms.Compose(
                [
                    transforms.CenterCrop(self.resolution) if CenterCrop else transforms.RandomCrop(self.resolution),
                    transforms.RandomHorizontalFlip() if Flip else transforms.Lambda(lambda x: x),
                ]
            )



    def __getitem__(self, index):

        if self.split == 'train':
            data_fld = self.train_data_list[index]

        if 'UltraEdit-Object' in data_fld:
            json_files = [f for f in os.listdir(data_fld) if f.startswith('task_info')]
            with open(os.path.join(data_fld, json_files[0]), 'r', encoding='utf-8') as f:
                task_json = json.load(f)
            instruction = task_json["instruction"]
            bbox = task_json["bbox_list"][0]
            polygon = task_json["polygon_list"][0]

            source_image_files = [f for f in os.listdir(data_fld) if f.startswith('source_image')]
            source_image = PIL.Image.open(os.path.join(data_fld, source_image_files[0]))
            ori_size = source_image.size
            source_image = crop_square_img(source_image, bbox, expand_rate=self.ExpRate, min_len=self.MinCrop)
            source_image = source_image.convert("RGB").resize((self.resolution, self.resolution))
            source_image = np.array(source_image).transpose(2, 0, 1)

            target_image_files = [f for f in os.listdir(data_fld) if f.startswith('edited_image')]
            target_image = PIL.Image.open(os.path.join(data_fld, target_image_files[0]))
            target_image = crop_square_img(target_image, bbox, expand_rate=self.ExpRate, min_len=self.MinCrop)
            target_image = target_image.convert("RGB").resize((self.resolution, self.resolution))
            target_image = np.array(target_image).transpose(2, 0, 1)

            mask = bbox_to_mask(bbox, ori_size)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            mask = Image.fromarray(mask)
            mask = crop_square_img(mask, bbox, expand_rate=self.ExpRate, min_len=self.MinCrop)
            mask = mask.convert("RGB").resize((self.resolution, self.resolution),resample=Image.NEAREST)
            mask = np.array(mask).transpose(2, 0, 1)

        
        else:
            json_files = [f for f in os.listdir(data_fld) if f.startswith('task_info')]
            with open(os.path.join(data_fld, json_files[0]), 'r', encoding='utf-8') as f:
                task_json = json.load(f)
            instruction = task_json["instruction"]
            bbox = task_json["bbox_list"][0]
            polygon = task_json["polygon_list"][0]

            source_image_files = [f for f in os.listdir(data_fld) if f.startswith('image_source')]
            source_image = PIL.Image.open(os.path.join(data_fld, source_image_files[0]))
            ori_size = source_image.size
            source_image = crop_square_img(source_image, bbox, expand_rate=self.ExpRate, min_len=self.MinCrop)
            source_image = source_image.convert("RGB").resize((self.resolution, self.resolution))
            source_image = np.array(source_image).transpose(2, 0, 1)

            target_image_files = [f for f in os.listdir(data_fld) if f.startswith('image_target')]
            target_image = PIL.Image.open(os.path.join(data_fld, target_image_files[0]))
            target_image = crop_square_img(target_image, bbox, expand_rate=self.ExpRate, min_len=self.MinCrop)
            target_image = target_image.convert("RGB").resize((self.resolution, self.resolution))
            target_image = np.array(target_image).transpose(2, 0, 1)

            mask = bbox_to_mask(bbox, ori_size)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            mask = Image.fromarray(mask)
            mask = crop_square_img(mask, bbox, expand_rate=self.ExpRate, min_len=self.MinCrop)
            mask = mask.convert("RGB").resize((self.resolution, self.resolution),resample=Image.NEAREST)
            mask = np.array(mask).transpose(2, 0, 1)

        images = np.concatenate([source_image, target_image, mask])
        images = torch.tensor(images)
        if self.split == 'train':
            images = self.train_transforms(images)
        source_image, target_image, mask = images.chunk(3)
        
        source_image = 2 * (source_image / 255) - 1
        source_image = source_image.reshape(3, self.resolution, self.resolution)
        target_image = 2 * (target_image / 255) - 1
        target_image = target_image.reshape(3, self.resolution, self.resolution)
        mask = mask / 255
        mask = mask.reshape(3, self.resolution, self.resolution)


        if self.split == 'train':
            return {
                'original_pixel_values': source_image, 
                'edited_pixel_values': target_image, 
                'mask_pixel_values': mask, 
                'edit_prompt': instruction,
                'name': data_fld.split('/')[-1]
                }
        if self.split == 'test':
            return {
                'original_pixel_values': source_image, 
                'edited_pixel_values': target_image, 
                'mask_pixel_values': mask, 
                'edit_prompt': instruction,
                'name': data_fld.split('/')[-1],
                'path': os.path.join(data_fld, source_image_files[0]),
                }

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.split == 'train':
            return len(self.train_data_list)
        if self.split == 'test':
            return len(self.test_data_list)
        





class CLIP_Metrics(nn.Module):
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder
        self.device = device

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(self.device)}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to(self.device)}

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(self, img_feat_one, img_feat_two, text_feat_one, text_feat_two):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one)
        return sim_direction

    def compute_image_similarity(self, img_feat_one, img_feat_two):
        sim_imgs = F.cosine_similarity(img_feat_two, img_feat_one)
        return sim_imgs

    def compute_text_image_alignment(self, img_feat_two, text_feat_two):
        alignment_text_image = F.cosine_similarity(img_feat_two, text_feat_two)
        return alignment_text_image

    def forward(self, image_one, image_two, caption_one, caption_two, image_gt):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        img_feat_gt = self.encode_image(image_gt)

        sim_directional = self.compute_directional_similarity(
            img_feat_one, img_feat_two, text_feat_one, text_feat_two
        )

        sim_images = self.compute_image_similarity(
            img_feat_one, img_feat_two
        )

        alilgnment_text_image = self.compute_text_image_alignment(
            img_feat_two, text_feat_two
        )

        sim_with_gt = self.compute_image_similarity(
            img_feat_two, img_feat_gt
        )

        return sim_directional, sim_images, alilgnment_text_image, sim_with_gt



#########################################################################################################################
#########################################################################################################################
#########################################################################################################################





def load_text_encoders(class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def tokenize_prompt(tokenizer, prompt, max_sequence_length=77):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
        text_encoder,
        tokenizer,
        max_sequence_length,
        text_encoder_dtype,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
        text_input_ids=None

):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    if text_input_ids is None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    prompt_embeds = prompt_embeds.to(dtype=text_encoder_dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
        text_encoder,
        tokenizer,
        prompt: str,
        text_encoder_dtype,
        device=None,
        num_images_per_prompt: int = 1,
        text_input_ids=None
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    if text_input_ids is None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder_dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
        text_encoders,
        tokenizers,
        prompt: str,
        max_sequence_length=None,
        text_encoders_dtypes=[torch.float32,torch.float32,torch.float32],
        device=None,
        num_images_per_prompt: int = 1,
        text_input_ids_list=None
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]
    clip_text_encoders_dtypes = text_encoders_dtypes[:2]
    if text_input_ids_list is not None:
        clip_text_input_ids_list = text_input_ids_list[:2]
    else:
        clip_text_input_ids_list = [None, None]
    zipped_text_encoders = zip(clip_tokenizers, clip_text_encoders, clip_text_encoders_dtypes, clip_text_input_ids_list)
    for tokenizer, text_encoder, clip_text_encoder_dtype, text_input_ids in zipped_text_encoders:
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            text_encoder_dtype=clip_text_encoder_dtype,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids,

        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    if text_input_ids_list is not None:
        t5_text_input_ids = text_input_ids_list[-1]
    else:
        t5_text_input_ids = None
    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        clip_text_encoders_dtypes[-1],
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
        text_input_ids=t5_text_input_ids
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "BleachNick/UltraEdit_500k": ("source_image", "edited_image", "edit_prompt"),
}
WANDB_TABLE_COL_NAMES = ["source_image", "edited_image", "edit_prompt"]


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--ori_model_name_or_path",
        type=str,
        default=None,
        help="Path to ori_model_name_or_path.",
    )
    parser.add_argument(
        "--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"]
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
             "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
             "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_jsonl",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--original_image_column",
        type=str,
        default="source_image",
        help="The column of the dataset containing the original image on which edits where made.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--edited_image_column",
        type=str,
        default="edited_image",
        help="The column of the dataset containing the edited image.",
    )
    parser.add_argument(
        "--edit_prompt_column",
        type=str,
        default="edit_prompt",
        help="The column of the dataset containing the edit instruction.",
    )
    parser.add_argument(
        "--val_image_url",
        type=str,
        default='images/input.png',
        help="URL to the original image that you would like to edit (used during inference for debugging purposes).",
    )
    parser.add_argument(
        '--val_mask_url',
        type=str,
        default='images/mask_img.png',
        help="URL to the mask image that you would like to edit (used during inference for debugging purposes).",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default="What if the horse wears a hat?", help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--validation_step",
        type=int,
        default=5000,
        help=(
            "Run fine-tuning validation every X steps. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--top_training_data_sample",
        type=int,
        default=None,
        help="Number of top samples to use for training, ranked by clip-sim-dit. If None, use the full dataset.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd3_edit",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='/15929303569/cache/',
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--eval_resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--do_mask", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--mask_column",
        type=str,
        default="mask_image",
        help="The column of the dataset containing the original image`s mask.",
    )

    parser.add_argument(
        "--ExpRate",
        type=float,
        default=1.2,
        help="Expand rate of cropping for training.",
    )

    parser.add_argument(
        "--MinCrop",
        type=int,
        default=192,
        help="The min length of crop.",
    )

    parser.add_argument(
        "--pretrained_editing_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_jsonl is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified

    return args


def combine_rgb_and_mask_to_rgba(rgb_image, mask_image):
    # Ensure the input images are the same size
    if rgb_image.size != mask_image.size:
        raise ValueError("The RGB image and the mask image must have the same dimensions")

    # Convert the mask image to 'L' mode (grayscale) if it is not
    if mask_image.mode != 'L':
        mask_image = mask_image.convert('L')

    # Split the RGB image into its three channels
    r, g, b = rgb_image.split()

    # Combine the RGB channels with the mask to form an RGBA image
    rgba_image = Image.merge("RGBA", (r, g, b, mask_image))

    return rgba_image


def convert_to_np(image, resolution):
    try:
        if isinstance(image, str):
            if image == "NONE":
                image = PIL.Image.new("RGB", (resolution, resolution), (255, 255, 255))
            else:
                image = PIL.Image.open(image)
        elif image is None:
            image = PIL.Image.new("RGB", (resolution, resolution), (255, 255, 255))
        image = image.convert("RGB").resize((resolution, resolution))
        return np.array(image).transpose(2, 0, 1)
    except Exception as e:
        print("Load error", image)
        print(e)
        # New blank image
        image = PIL.Image.new("RGB", (resolution, resolution), (255, 255, 255))
        return np.array(image).transpose(2, 0, 1)


def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def main():
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    
    if args.resume_from_checkpoint is not None:
        args.output_dir = args.resume_from_checkpoint


    import datetime
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=999999999))

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    from accelerate import DistributedDataParallelKwargs as DDPK 
    kwargs = DDPK(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    def download_image(path_or_url,resolution=512):
        # Check if path_or_url is a local file path
        if path_or_url is None:
            # return a white RBG image image
            return PIL.Image.new("RGB", (resolution, resolution), (255, 255, 255))
        if os.path.exists(path_or_url):
            image = Image.open(path_or_url).convert("RGB").resize((resolution, resolution))

        else:
            image = Image.open(requests.get(path_or_url, stream=True).raw).convert("RGB")

        image = PIL.ImageOps.exif_transpose(image)
        return image

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.ERROR,         # DEBUG INFO WARNING   ERROR  CRITICAL
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    # transformer = SD3Transformer2DModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant, 
    #     strict=False,
    #     low_cpu_mem_usage=False,
    #     device_map=None,  
    # )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_editing_model_path, subfolder="transformer", revision=args.revision, variant=args.variant, 
        strict=False,
        low_cpu_mem_usage=False,
        device_map=None,  
    )


    print('-------------------------')
    print('meta parameters: ')
    for name, param in transformer.named_parameters():
        if param.is_meta:
            print(name)
    print('-------------------------')


    # ============================================================
    # ✅ moe_out_proj + moe_gating_temb
    # ============================================================
    for name, module in transformer.named_modules():
        if name.endswith("moe_out_proj") or name.endswith("moe_gating_temb"):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

                print(
                    f"[Init] {name} ({module.in_features}->{module.out_features}) -> zeros"
                )

    with torch.no_grad():
        for name, param in transformer.named_parameters():
            if name.endswith("moe_out_proj.weight") or name.endswith("moe_gating_temb.weight"):
                print(name, param.abs().sum().item())

    print('transformer.config.in_channels = ', transformer.config.in_channels)      # transformer.config.in_channels =  48


    # ============================================================
    # moe
    # ============================================================
    transformer.requires_grad_(False)
    for name, param in transformer.named_parameters():
        if (
            "moe_experts" in name
            or "moe_gating" in name
            or "moe_gating_temb" in name
            or "moe_out_proj" in name
        ):
            param.requires_grad = True
    # ============================================================

    vae.requires_grad_(False)
    if args.train_text_encoder:
        text_encoder_one.requires_grad_(True)
        text_encoder_two.requires_grad_(True)
        text_encoder_three.requires_grad_(True)
    else:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        text_encoder_three.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=torch.float32)

    if not args.train_text_encoder:
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
        text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()
            text_encoder_three.gradient_checkpointing_enable()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(unwrap_model(model), SD3Transformer2DModel):
                    unwrap_model(model).save_pretrained(os.path.join(output_dir, "transformer"))
                elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
                    if isinstance(unwrap_model(model), CLIPTextModelWithProjection):
                        hidden_size = unwrap_model(model).config.hidden_size
                        if hidden_size == 768:
                            unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder"))
                        elif hidden_size == 1280:
                            unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_2"))
                    else:
                        unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_3"))
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if isinstance(unwrap_model(model), SD3Transformer2DModel):
                print('=====================')
                print(input_dir)
                load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
            elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
                try:
                    load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder")
                    model(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                except Exception:
                    try:
                        load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder_2")
                        model(**load_model.config)
                        model.load_state_dict(load_model.state_dict())
                    except Exception:
                        try:
                            load_model = T5EncoderModel.from_pretrained(input_dir, subfolder="text_encoder_3")
                            model(**load_model.config)
                            model.load_state_dict(load_model.state_dict())
                        except Exception:
                            raise ValueError(f"Couldn't load the model of type: ({type(model)}).")
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")

            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # transformer_parameters_with_lr = {"params": transformer.parameters(), "lr": args.learning_rate}
    # ============================================================
    # moe
    # ============================================================
    transformer_parameters_with_lr = {
        "params": [p for p in transformer.parameters() if p.requires_grad],
        "lr": args.learning_rate,
    }


    num_params = sum(p.numel() for p in transformer_parameters_with_lr["params"])
    num_params_m = num_params / 1e6
    print(f"[Optimizer] Trainable params (requires_grad=True): {num_params_m:.3f} M")   # [Optimizer] Trainable params (requires_grad=True): 61.637 M
    assert num_params > 0, (
        "❌ No trainable parameters found! "
        "Did you forget to set requires_grad=True for MoE modules?"
    )

    trainable = [n for n, p in transformer.named_parameters() if p.requires_grad]
    frozen = [n for n, p in transformer.named_parameters() if not p.requires_grad]

    print(f"[Params] Trainable: {len(trainable)}, Frozen: {len(frozen)}")

    print("[Trainable params]")
    for n in trainable:
        print(f"  {n}")  
    # ============================================================

    if args.train_text_encoder:
        # different learning rate for text encoder and unet
        text_parameters_one_with_lr = {
            "params": text_encoder_one.parameters(),
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        text_parameters_two_with_lr = {
            "params": text_encoder_two.parameters(),
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        text_parameters_three_with_lr = {
            "params": text_encoder_three.parameters(),
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [
            transformer_parameters_with_lr,
            text_parameters_one_with_lr,
            text_parameters_two_with_lr,
            text_parameters_three_with_lr,
        ]
    else:
        params_to_optimize = [transformer_parameters_with_lr]

    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    # Initialize the optimizer
    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate
            params_to_optimize[2]["lr"] = args.learning_rate
            params_to_optimize[3]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )
    text_encoders_dtypes = [text_encoder_one.dtype, text_encoder_two.dtype, text_encoder_three.dtype]
    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
        text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
        def compute_text_embeddings(prompt, text_encoders, tokenizers,text_encoders_dtypes):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length, text_encoders_dtypes
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).



    from torch.utils.data import DataLoader

    edit_prompt_column = args.edit_prompt_column

    train_dataset = IIE_Dataset(split='train', ExpRate=args.ExpRate, MinCrop=args.MinCrop)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers,)
    print('train dataset containing ', len(train_dataloader), 'images')



    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            transformer,
            text_encoder_one,
            text_encoder_two,
            text_encoder_three,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            text_encoder_one,
            text_encoder_two,
            text_encoder_three,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.


    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    print('=========num_update_steps_per_epoch==========', num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("instruct-pix2pix_sd3", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            # path = os.path.basename(args.resume_from_checkpoint)
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            resume_global_step = global_step * args.gradient_accumulation_steps
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    else:
        initial_global_step = 0
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(0, args.max_train_steps), initial=initial_global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma


    # with torch.autograd.set_detect_anomaly(True):
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()
            text_encoder_three.train()

        train_loss = 0.0
        for iter_step, batch in enumerate(train_dataloader):

            models_to_accumulate = [transformer]
            if args.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one, text_encoder_two, text_encoder_three])
            with accelerator.accumulate(models_to_accumulate):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.]
                pixel_values = batch["edited_pixel_values"].to(dtype=vae.dtype)
                prompt = batch[edit_prompt_column]

                if not args.train_text_encoder:
                    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                        prompt, text_encoders, tokenizers,text_encoders_dtypes
                    )
                else:
                    tokens_one = tokenize_prompt(tokenizer_one, prompt)
                    tokens_two = tokenize_prompt(tokenizer_two, prompt)
                    tokens_three = tokenize_prompt(tokenizer_three, prompt, args.max_sequence_length)

                latents = vae.encode(pixel_values).latent_dist.sample() # print(latents.shape)  # torch.Size([8, 16, 64, 64])
                latents = latents * vae.config.scaling_factor
                latents = latents.to(dtype=weight_dtype)


                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                if args.weighting_scheme == "logit_normal":
                    # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
                    u = torch.normal(mean=args.logit_mean, std=args.logit_std, size=(bsz,), device="cpu")
                    u = torch.nn.functional.sigmoid(u)
                elif args.weighting_scheme == "mode":
                    u = torch.rand(size=(bsz,), device="cpu")
                    u = 1 - u - args.mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
                else:
                    u = torch.rand(size=(bsz,), device="cpu")

                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                original_image_embeds = vae.encode(batch["original_pixel_values"].to(vae.dtype)).latent_dist.mode()
                concatenated_noisy_latents = torch.cat([noisy_model_input, original_image_embeds], dim=1)

                if args.do_mask:
                    mask_embeds = vae.encode(batch["mask_pixel_values"].to(vae.dtype)).latent_dist.mode()
                    concatenated_noisy_latents = torch.cat([concatenated_noisy_latents, mask_embeds], dim=1)

                # mask_values = batch["mask_pixel_values"].to(dtype=vae.dtype)

                # Predict the noise residual
                if not args.train_text_encoder:
                    model_pred = transformer(
                        hidden_states=concatenated_noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                else:
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two, text_encoder_three],
                        tokenizers=[tokenizer_one, tokenizer_two, tokenizer_three],
                        prompt=prompt,
                        text_input_ids_list=[tokens_one, tokens_two, tokens_three],
                        max_sequence_length=args.max_sequence_length,
                        text_encoders_dtypes = text_encoders_dtypes
                    )

                    model_pred = transformer(
                        hidden_states=concatenated_noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                        # mask_index=mask_index,
                    )[0]

                model_pred = model_pred * (-sigmas) + noisy_model_input
                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                if args.weighting_scheme == "sigma_sqrt":
                    weighting = (sigmas ** -2.0).float()
                elif args.weighting_scheme == "cosmap":
                    bot = 1 - 2 * sigmas + 2 * sigmas ** 2
                    weighting = 2 / (math.pi * bot)
                else:
                    weighting = torch.ones_like(sigmas)

                target = latents
                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.

                # Concatenate the `original_image_embeds` with the `noisy_latents`.

                # Get the target for loss depending on the prediction type
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(
                            transformer.parameters(),
                            text_encoder_one.parameters(),
                            text_encoder_two.parameters(),
                            text_encoder_three.parameters(),
                        )
                        if args.train_text_encoder
                        else transformer.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()



            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break


    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()

    main()

