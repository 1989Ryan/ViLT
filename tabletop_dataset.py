import torch
import os
import numpy as np
from PIL import Image
import json
from torchvision.transforms import functional as F
from torchvision.io import read_image
import random
from vilt.transforms import pixelbert_transform
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer
# import cv2
# from tabletop_gym.envs.data.template import instruction_template
# import math

def read_json(filepath):
    '''
    from filepath to instruction list
    :return:instruction list
    '''
    try:
        with open(filepath) as f:
            data = json.load(f)
    except IOError as exc:
        raise IOError("%s: %s" % (filepath, exc.strerror))
    return data

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

class tabletop_gym_objpick_dataset(torch.utils.data.Dataset):
    def __init__(self, _config, root, num=None, test=False, device=None):
        self.root = root
        self.class_num = 19
        self.device = device
        self.tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
        # load all image files, sorting them to
        # ensure that they are aligned
        if not test:
            # list_dir_1 = listdir_fullpath(root + '/train_4_obj_nvisii_pick') 
            # list_dir_2 = listdir_fullpath(root + '/train_10_obj_nvisii')
            list_dir = listdir_fullpath(root + '/train_11_obj_nvisii_pick')
        else:
            list_dir = listdir_fullpath(root + "/test_11_obj_nvisii_pick") 
        self.test = test
        if num is not None:
            paths = random.sample(list_dir, int(num)) 
        else:
            paths = list(sorted(list_dir))
        info_simple = [read_json(os.path.join(ele, "info_simple.json")) for ele in paths]
        info_compositional = [read_json(os.path.join(ele, "info_compositional.json")) for ele in paths]
        self.paths = paths + paths
        self.info = info_simple + info_compositional
    
    def __getitem__(self, idx):
        '''
        in the dataset we need to predefine several components
        '''
        # load images and masks
        img_path = os.path.join(self.paths[idx], "rgb.png")
        mask_path = os.path.join(self.paths[idx], "mask.png")
        # info_path = os.path.join(self.paths[idx], "info_simple.json")
        # info_comp_path = os.path.join(self.paths[idx], "info_compositional.json")

        info = self.info[idx]
        ins = info['instruction']
        # simple_ins = info['simple_instruction']
        # complex_ins = info['complex_instruction']
        # img = read_image(img_path)
        # print(img)
        # mask = read_image(mask_path)

        t_bbox = info['target_bbox']
        t_pixel = torch.tensor([
            t_bbox[0] + t_bbox[2],
            t_bbox[1] + t_bbox[3],
        ]).to(self.device)/2
        image = Image.open(img_path).convert("RGB")
        img = pixelbert_transform(size=384)(image)
        sample = {}
       
        # sample["masks"] = mask
        sample['pick_labels']= t_pixel
        sample['place_labels']= t_pixel
        # sample['label_place_2'] = info_comp['goal_pixel']
        # sample['bboxes'] = info['bboxes']
        # sample['target_bbox'] = info['target_bbox']
        # if self.test:
        #     sample['relations_1'] = info["relations"]
        #     sample['relations_2'] = info_comp["relations"]
        encoded = self.tokenizer(ins,
            padding="max_length",
            truncation=True,
            max_length=40,
            return_special_tokens_mask=True,
                                 )
        sample["text"] = (ins, encoded)
        sample["image"] = [img]
        # sample['raw_img'] = img_path
        sample['text_ids']=torch.tensor(encoded["input_ids"]).to(self.device)
        sample["text_labels"] = torch.tensor(encoded["input_ids"]).to(self.device)
        sample["text_masks"] = torch.tensor(encoded["attention_mask"]).to(self.device)
        return sample

    def __len__(self):
        return len(self.paths)

class tabletop_gym_obj_dataset(torch.utils.data.Dataset):
    '''object detector dataset of tabletop gym'''
    def __init__(self, _config, root, num=None, test=False, device=None):
        self.root = root
        self.class_num = 19
        self.device = device
        self.tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
        # load all image files, sorting them to
        # ensure that they are aligned
        if not test:
            list_dir_1 = listdir_fullpath(root + '/train_4_obj_nvisii') 
            list_dir_2 = listdir_fullpath(root + '/train_10_obj_nvisii')
            list_dir_3 = listdir_fullpath(root + '/train_11_obj_nvisii')
        else:
            list_dir = listdir_fullpath(root + "/test_11_obj_nvisii") 
        self.test = test
        if num is not None:
            paths = random.sample(list_dir_1, int(num)) \
                + random.sample(list_dir_2, int(num)) \
                + random.sample(list_dir_3, int(num))
        else:
            paths = list(sorted(list_dir))
        info_simple = [read_json(os.path.join(ele, "info_simple.json")) for ele in paths]
        info_compositional = [read_json(os.path.join(ele, "info_compositional.json")) for ele in paths]
        self.paths = paths + paths
        self.info = info_simple + info_compositional
    
    def __getitem__(self, idx):
        '''
        in the dataset we need to predefine several components
        '''
        # load images and masks
        img_path = os.path.join(self.paths[idx], "rgb.png")
        mask_path = os.path.join(self.paths[idx], "mask.png")
        # info_path = os.path.join(self.paths[idx], "info_simple.json")
        # info_comp_path = os.path.join(self.paths[idx], "info_compositional.json")

        info = self.info[idx]
        ins = info['instruction']
        # simple_ins = info['simple_instruction']
        # complex_ins = info['complex_instruction']
        # img = read_image(img_path)
        # print(img)
        mask = read_image(mask_path)

        image = Image.open(img_path).convert("RGB")
        img = pixelbert_transform(size=384)(image)
        # img = img.unsqueeze(0).to(self.device)
        # img = F.pil_to_tensor(img)
        sample = {}
       
        # sample["masks"] = mask
        sample['place_labels']= info['goal_pixel']
        sample['pick_labels']= info['goal_pixel']
        # sample['label_place_2'] = info_comp['goal_pixel']
        # sample['bboxes'] = info['bboxes']
        # sample['target_bbox'] = info['target_bbox']
        # if self.test:
        #     sample['relations_1'] = info["relations"]
        #     sample['relations_2'] = info_comp["relations"]
        encoded = self.tokenizer(ins,
            padding="max_length",
            truncation=True,
            max_length=40,
            return_special_tokens_mask=True,
                                 )
        sample["text"] = (ins, encoded)
        sample["image"] = [img]
        # sample['raw_img'] = img_path
        sample['text_ids']=torch.tensor(encoded["input_ids"]).to(self.device)
        sample["text_labels"] = torch.tensor(encoded["input_ids"]).to(self.device)
        sample["text_masks"] = torch.tensor(encoded["attention_mask"]).to(self.device)
        return sample

    def __len__(self):
        return len(self.paths)