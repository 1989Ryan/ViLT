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
import cv2
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
    '''object detector dataset of tabletop gym'''
    def __init__(self, root, num=None, test=False):
        self.root = root
        self.class_num = 19
        # load all image files, sorting them to
        # ensure that they are aligned
        if not test:
            list_dir_3 = listdir_fullpath(root + '/train_11_obj_nvisii_pick')
        else:
            list_dir = listdir_fullpath(root) 
        self.test = test
        if num is not None:
            self.paths = random.sample(list_dir_3, int(num)) 
        else:
            self.paths = list(sorted(list_dir))
        self.info_simple = [read_json(os.path.join(ele, "info_simple.json")) for ele in self.paths]
        self.info_compositional = [read_json(os.path.join(ele, "info_compositional.json")) for ele in self.paths]

    
    def __getitem__(self, idx):
        '''
        in the dataset we need to predefine several components
        '''
        # load images and masks
        img_path = os.path.join(self.paths[idx], "rgb.png")
        mask_path = os.path.join(self.paths[idx], "mask.png")
        # info_path = os.path.join(self.paths[idx], "info_simple.json")
        # info_comp_path = os.path.join(self.paths[idx], "info_compositional.json")

        info = self.info_simple[idx]
        info_comp = self.info_compositional[idx]
        ins = info['instruction']
        ins_comp = info_comp['instruction']
        # simple_ins = info['simple_instruction']
        # complex_ins = info['complex_instruction']
        # img = read_image(img_path)
        # print(img)
        mask = read_image(mask_path)

        img = Image.open(img_path).convert("RGB")
        img = F.pil_to_tensor(img)
        sample = {}
       
        sample["masks"] = mask
        # sample['label_place']= info['goal_pixel']
        # sample['label_place_2'] = info_comp['goal_pixel']
        sample['bboxes'] = info['bboxes']
        sample['target_bbox'] = info['target_bbox']
        if self.test:
            sample['relations_1'] = info["relations"]
            sample['relations_2'] = info_comp["relations"]
        sample["ins_1"] = ins
        sample["ins_2"] = ins_comp
        sample["image"] = F.convert_image_dtype(img)
        sample['raw_img'] = img_path
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
    
    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i]

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]
            view_size = len(img[0])

            new_images = [
                torch.zeros(batch_size, 3, max_height, max_width)
                for _ in range(view_size)
            ]

            for bi in range(batch_size):
                orig_batch = img[bi]
                for vi in range(view_size):
                    if orig_batch is None:
                        new_images[vi][bi] = None
                    else:
                        orig = img[bi][vi]
                        new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

            dict_batch[img_key] = new_images

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

        return dict_batch