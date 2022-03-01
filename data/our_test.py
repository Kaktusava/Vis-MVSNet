from utils.io_utils import load_cam, load_pfm, cam_adjust_max_d
from utils.preproc import to_channel_first, resize, bottom_left_crop, center_crop, image_net_center as center_image
from data.data_utils import dict_collate


import json
import os
from itertools import accumulate

import cv2
import numpy as np
import torch.utils.data as data

from utils.preproc import image_net_center as center_image, mask_depth_image, to_channel_first, resize, bottom_left_crop, random_crop, center_crop, recursive_apply
from utils.preproc import random_brightness, random_contrast, motion_blur
from utils.io_utils import load_cam, load_pfm
from data.data_utils import dict_collate, Until, Cycle
import PIL


def load_pair(file: str):
    with open(file) as f:
        lines = f.readlines()
    n_cam = int(lines[0])
    pairs = {}
    img_ids = []
    for i in range(1, 1+2*n_cam, 2):
        pair = []
        score = []
        img_id = lines[i].strip()
        pair_str = lines[i+1].strip().split(' ')
        n_pair = int(pair_str[0])
        for j in range(1, 1+2*n_pair, 2):
            pair.append(pair_str[j])
            score.append(float(pair_str[j+1]))
        img_ids.append(img_id)
        pairs[img_id] = {'id': img_id, 'index': i//2, 'pair': pair, 'score': score}
    pairs['id_list'] = img_ids
    return pairs

class MyDataset(data.Dataset):

    def __init__(self, root, list_dir, list_file, num_src, read, transforms):
        super().__init__()
        self.root = root
        self.list_dir = list_dir
        self.num_src = num_src
        self.scene_list = []
        self.light_list = []
        with open(os.path.join(list_dir, list_file)) as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    self.scene_list.append(elems[0])
                    self.light_list.append(elems[1])

        self.pair_list = [
            load_pair(os.path.join(list_dir, scene, 'pair.txt'))
            for scene in self.scene_list
            ]
        self.index2scene = [[(i, j) for j in range(len(self.pair_list[i]['id_list']))] for i in range(len(self.scene_list))]
        self.index2scene = sum(self.index2scene, [])
        self.read = read
        self.transforms = transforms
        print(f'Number of samples: {len(self.index2scene)}')

        
    def _idx2filename(self, scene_idx, img_id, file_type):
        if img_id == 'dummy': return 'dummy'
        depth_id = img_id.zfill(4)
        cam_id = img_id.zfill(8)
        img_id = img_id.zfill(4)
        
        if file_type == 'img':
            return os.path.join(self.root, 'processed_scans', 'images/undist', self.scene_list[scene_idx],'tis_right/rgb', self.light_list[scene_idx], f'{img_id}.png')
        if file_type == 'cam':
            return os.path.join(self.list_dir, self.scene_list[scene_idx], 'cams', f'{cam_id}_cam.txt')    
        
    def __len__(self):
        return len(self.index2scene)

    def __getitem__(self, i):
        scene_idx, ref_idx = self.index2scene[i]
        ref_id = self.pair_list[scene_idx]['id_list'][ref_idx]
        # print(ref_id)
        skip = 0
        if len(self.pair_list[scene_idx][ref_id]['pair']) < self.num_src:
            skip = 1
            print(f'sample {i} does not have enough sources')
        src_ids = self.pair_list[scene_idx][ref_id]['pair'][:self.num_src]
        if skip: src_ids += ['dummy'] * (self.num_src - len(src_ids))
        ref = self._idx2filename(scene_idx, ref_id, 'img')
        # print(ref)
        ref_cam = self._idx2filename(scene_idx, ref_id, 'cam')
        srcs = [self._idx2filename(scene_idx, src_id, 'img') for src_id in src_ids]
        srcs_cam = [self._idx2filename(scene_idx, src_id, 'cam') for src_id in src_ids]
        filenames = {
            'ref': ref, 
            'ref_cam': ref_cam, 
            'srcs': srcs, 
            'srcs_cam': srcs_cam, 
            'skip': skip
        }

        sample = self.read(filenames)
        for transform in self.transforms:
            sample = transform(sample)
        return sample

def read(filenames, max_d, interval_scale):
    ref_name, ref_cam_name, srcs_name, srcs_cam_name, skip = [filenames[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'skip']]
    ref, *srcs = [cv2.imread(fn) if fn != 'dummy' else None for fn in [ref_name] + srcs_name]
    srcs = [src if src is not None else np.ones_like(ref, dtype=np.uint8) for src in srcs]
    ref_cam, *srcs_cam = [load_cam(fn, max_d, interval_scale) if fn != 'dummy' else None for fn in [ref_cam_name] + srcs_cam_name]
    srcs_cam = [src_cam if src_cam is not None else np.ones_like(ref_cam, dtype=np.float32) for src_cam in srcs_cam]
    # gt = np.expand_dims(load_pfm(gt_name), -1)
    gt = np.zeros((ref.shape[0], ref.shape[1], 1))
    masks = [np.zeros((ref.shape[0], ref.shape[1], 1)) for i in range(len(srcs))]
    if ref_cam[1,3,0] <= 0:
        skip = 1
        print(f'depth start <= 0')
    return {
        'ref': ref,
        'ref_cam': ref_cam,
        'srcs': srcs,
        'srcs_cam': srcs_cam,
        'gt': gt,
        'masks': masks,
        'skip': skip
    }
   

def val_preproc(sample, preproc_args):
    ref, ref_cam, srcs, srcs_cam, gt, masks, skip = [sample[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'gt', 'masks', 'skip']]

    ref, *srcs = [center_image(img) for img in [ref] + srcs]
    ref, ref_cam, srcs, srcs_cam, gt, masks = resize([ref, ref_cam, srcs, srcs_cam, gt, masks], preproc_args['resize_width'], preproc_args['resize_height'])
    ref, ref_cam, srcs, srcs_cam, gt, masks = bottom_left_crop([ref, ref_cam, srcs, srcs_cam, gt, masks], preproc_args['crop_width'], preproc_args['crop_height'])
    ref, *srcs, gt = to_channel_first([ref] + srcs + [gt])
    masks = to_channel_first(masks)

    srcs, srcs_cam, masks = [np.stack(arr_list, axis=0) for arr_list in [srcs, srcs_cam, masks]]

    return {
        'ref': ref,  # 3hw
        'ref_cam': ref_cam,  # 244
        'srcs': srcs,  # v3hw
        'srcs_cam': srcs_cam,  # v244
        'gt': gt,  # 1hw
        'masks': masks,  # v1hw
        'skip': skip  # scalar
    }


def get_val_loader(root, list_dir, part, num_src, preproc_args):
    dataset = MyDataset(
        root, list_dir, f'test_list_{part}.txt', num_src,
        read=lambda filenames: read(filenames, preproc_args['max_d'], preproc_args['interval_scale']),
        transforms=[lambda sample: val_preproc(sample, preproc_args)]
    )
    loader = data.DataLoader(dataset, 1, collate_fn=dict_collate, shuffle=False)
    return dataset, loader
