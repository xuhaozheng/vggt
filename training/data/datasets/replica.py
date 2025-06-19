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
from numpy.linalg import inv
# from vggt.training.data.dataset_util import *
# from vggt.training.data.base_dataset import BaseDataset

from training.data.base_dataset import BaseDataset
from training.data.dataset_util import *

class ReplicaDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        DATA_DIR: str = None,
        DATA_ANNOTATION_DIR: str = None,
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        img_h: int =480, 
        img_w: int =640,
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

        if DATA_DIR is None is None:
            raise ValueError("Both PD_DIR and PD_ANNOTATION_DIR must be specified.")

        self.dataset_label = 'replica'
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

        self.subdirs = []
        self.sequences = []
        self.seq_names = []

        ### Manual defined dataset parameters
        quick = True
        self.img_h = img_h
        self.img_w = img_w


        ### From replica loarder
        data_dir = DATA_DIR
        traj_file = os.path.join(data_dir, "traj_w_c.txt")
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.depth_dir = os.path.join(data_dir, "depth")  # depth is in mm uint
        self.semantic_class_dir = os.path.join(data_dir, "semantic_class")
        self.semantic_instance_dir = os.path.join(data_dir, "semantic_instance")
        if not os.path.exists(self.semantic_instance_dir):
            self.semantic_instance_dir = None

        # self.img_h = img_h
        # self.img_w = img_w

        self.Ts_full = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)

        self.rgb_list = sorted(glob.glob(self.rgb_dir + '/rgb*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        self.depth_list = sorted(glob.glob(self.depth_dir + '/depth*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        self.semantic_list = sorted(glob.glob(self.semantic_class_dir + '/semantic_class_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        if self.semantic_instance_dir is not None:
            self.instance_list = sorted(glob.glob(self.semantic_instance_dir + '/semantic_instance_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))

        self.num_samples = len(self.rgb_list)
        
        self.set_params_replica()


        ### VGGT dataset dict
        self.data_store = {}
        total_frame_num = 0
        self.data_store['Sequence2'] = []

        for idx in range(len(self.rgb_list)):
            anno = {}
            anno['filepath'] = self.rgb_list[idx]
            anno['depth'] = self.depth_list[idx]
            anno['extri'] = self.Ts_full[idx]
            anno['intri'] = self.K
            self.data_store['Sequence2'].append(anno)
            total_frame_num += 1

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Test"
        logging.info(f"{status}: PointOdyssey Data size: {self.sequence_list_len}")
        # logging.info(f"{status}: PointOdyssey Data length of training set: {len(self)}")

    def set_params_replica(self):
            self.H = self.img_h 
            self.W = self.img_w

            self.n_pix = self.H * self.W
            self.aspect_ratio = self.W/self.H

            self.hfov = 90
            # the pin-hole camera has the same value for fx and fy
            self.fx = self.W / 2.0 / math.tan(math.radians(self.hfov / 2.0))
            # self.fy = self.H / 2.0 / math.tan(math.radians(self.yhov / 2.0))
            self.fy = self.fx
            self.cx = (self.W - 1.0) / 2.0
            self.cy = (self.H - 1.0) / 2.0
            self.K = np.array([[self.fx, 0, self.cx],
                            [0, self.fy, self.cy],
                            [0, 0, 1]], dtype=np.float32)
            self.K_inv = np.linalg.inv(self.K)

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

        image_paths = []

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
            depth_map = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED) / 1000.0  # uint16 mm depth, then turn depth from mm to meter

            extri_opencv = extrinsic
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

            image_paths.append(impath)

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
            "image_paths": image_paths,
        }
        return batch