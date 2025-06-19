# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os.path as osp
import os
import logging
import glob
import cv2
import random
import numpy as np
import sys

# from vggt.training.data.dataset_util import *
# from vggt.training.data.base_dataset import BaseDataset

from training.data.base_dataset import BaseDataset
from training.data.dataset_util import *

class PointOdysseyDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        PD_DIR: str = None,
        PD_ANNOTATION_DIR: str = None,
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the Co3dDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            CO3D_DIR (str): Directory path to CO3D data.
            CO3D_ANNOTATION_DIR (str): Directory path to CO3D annotations.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If CO3D_DIR or CO3D_ANNOTATION_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random

        if PD_DIR is None or PD_ANNOTATION_DIR is None:
            raise ValueError("Both PD_DIR and PD_ANNOTATION_DIR must be specified.")
        '''
        Defalt values from pointodysseyDUST3R
        '''
        self.dataset_label = 'pointodyssey'
        self.split = split
        self.S = 2 # stride
        self.N = 16 # min num points
        self.verbose = False

        self.use_augs = False
        self.dset = split

        self.rgb_paths = []
        self.depth_paths = []
        self.normal_paths = []
        self.traj_paths = []
        self.annotation_paths = []
        self.full_idxs = []
        self.sample_stride = []
        self.strides = [1,2,3,4,5,6,7,8,9]

        self.subdirs = []
        self.sequences = []
        self.seq_names = []
        self.subdirs.append(os.path.join(PD_DIR, split))
        ### Manual defined dataset parameters
        quick = True
        clip_step_last_skip = 0
        clip_step = 2
        dist_type='linear_1_2'
        dataset_location = PD_DIR
        dset = split

        ### VGGT dataset dict
        self.data_store = {}

        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*/")):
                seq_name = seq.split('/')[-2]
                self.sequences.append(seq)
                self.seq_names.append(seq_name)

        self.sequences = sorted(self.sequences)
        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))
        ## load trajectories
        print('loading trajectories...')

        if quick:
           self.sequences = self.sequences[1:2] 


        total_frame_num = 0

        # for seq_name, seq_data in annotation.items():
        #     if len(seq_data) < min_num_images:
        #         continue
        #     if seq_name in self.invalid_sequence:
        #         continue
        #     total_frame_num += len(seq_data)
        #     self.data_store[seq_name] = seq_data

        for seq, seq_name in zip(self.sequences, self.seq_names):
            if self.verbose: 
                print('seq', seq)

            rgb_path = os.path.join(seq, 'rgbs')
            info_path = os.path.join(seq, 'info.npz')
            annotations_path = os.path.join(seq, 'anno.npz')
            
            if os.path.isfile(info_path) and os.path.isfile(annotations_path):

                info = np.load(info_path, allow_pickle=True)
                trajs_3d_shape = info['trajs_3d'].astype(np.float32)

                annotations_path = os.path.join(seq, 'anno.npz')
                annotations = np.load(annotations_path, allow_pickle=True)
                pix_T_cams = annotations['intrinsics'].astype(np.float32)
                world_T_cams = annotations['extrinsics'].astype(np.float32)

                if len(trajs_3d_shape) and trajs_3d_shape[1] > self.N:
                    self.data_store[seq_name] = []


                    total_frame_num += len(os.listdir(rgb_path))
                    print('num rgb',len(os.listdir(rgb_path)))
                    print('total_frame_num',total_frame_num)
                    self.annotation_paths.append(os.path.join(seq, 'anno.npz'))

                    for idx in range(len(os.listdir(rgb_path))):
                        anno = {}
                        anno['filepath'] = os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % idx)
                        anno['depth'] = os.path.join(seq, 'depths', 'depth_%05d.png' % idx)
                        anno['extri'] = world_T_cams[idx]
                        anno['intri'] = pix_T_cams[idx]        
                        self.data_store[seq_name].append(anno)         

                elif self.verbose:
                    print('rejecting seq for missing 3d')
            elif self.verbose:
                print('rejecting seq for missing info or anno')

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Test"
        logging.info(f"{status}: PointOdyssey Data size: {self.sequence_list_len}")
        # logging.info(f"{status}: PointOdyssey Data length of training set: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        metadata = self.data_store[seq_name]

        if ids is None:
            ids = np.random.choice(
                len(metadata), img_per_seq, replace=self.duplicate_img
            )

        annos = [metadata[i] for i in ids]

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        for anno in annos:   
            impath = anno['filepath']
            depthpath = anno['depth'] 

            # load camera params
            extrinsic = anno['extri']
            R = extrinsic[:3,:3]
            t = extrinsic[:3,3]
            camera_pose = np.eye(4, dtype=np.float32) # cam_2_world
            camera_pose[:3,:3] = R.T
            camera_pose[:3,3] = -R.T @ t
            intrinsic = anno['intri']

            # load image and depth
            image = imread_cv2(impath)
            depth16 = cv2.imread(depthpath, cv2.IMREAD_ANYDEPTH)
            depth_map = depth16.astype(np.float32) / 65535.0 * 1000.0 # 1000 is the max depth in the dataset

            extri_opencv = camera_pose
            intri_opencv = intrinsic

            original_size = np.array(image.shape[:2])


            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=impath,
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(impath)
            original_sizes.append(original_size)

        set_name = "co3d"

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch