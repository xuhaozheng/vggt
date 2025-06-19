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

import cv2
import random
import numpy as np


# from vggt.training.data.dataset_util import *
# from vggt.training.data.base_dataset import BaseDataset

from training.data.base_dataset import BaseDataset
from training.data.dataset_util import *
from torchvision import transforms as TF

# SEEN_CATEGORIES = [
#     "apple",
#     "backpack",
#     "banana",
#     "baseballbat",
#     "baseballglove",
#     "bench",
#     "bicycle",
#     "bottle",
#     "bowl",
#     "broccoli",
#     "cake",
#     "car",
#     "carrot",
#     "cellphone",
#     "chair",
#     "cup",
#     "donut",
#     "hairdryer",
#     "handbag",
#     "hydrant",
#     "keyboard",
#     "laptop",
#     "microwave",
#     "motorcycle",
#     "mouse",
#     "orange",
#     "parkingmeter",
#     "pizza",
#     "plant",
#     "stopsign",
#     "teddybear",
#     "toaster",
#     "toilet",
#     "toybus",
#     "toyplane",
#     "toytrain",
#     "toytruck",
#     "tv",
#     "umbrella",
#     "vase",
#     "wineglass",
# ]

SEEN_CATEGORIES = [
    "apple",
]


class Co3dDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        CO3D_DIR: str = None,
        CO3D_ANNOTATION_DIR: str = None,
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

        self.to_tensor = TF.ToTensor()

        if CO3D_DIR is None or CO3D_ANNOTATION_DIR is None:
            raise ValueError("Both CO3D_DIR and CO3D_ANNOTATION_DIR must be specified.")

        category = sorted(SEEN_CATEGORIES)

        self.duplicate_img = False
        self.mask_depth = True

        if split == "train":
            split_name_list = ["train"]
            self.len_train = len_train
        elif split == "test":
            split_name_list = ["test"]
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.invalid_sequence = [] # set any invalid sequence names here


        self.category_map = {}
        self.data_store = {}
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"CO3D_DIR is {CO3D_DIR}")

        self.CO3D_DIR = CO3D_DIR
        self.CO3D_ANNOTATION_DIR = CO3D_ANNOTATION_DIR

        total_frame_num = 0

        for c in category:
            for split_name in split_name_list:
                annotation_file = osp.join(
                    self.CO3D_ANNOTATION_DIR, f"{c}_{split_name}.jgz"
                )

                try:
                    with gzip.open(annotation_file, "r") as fin:
                        annotation = json.loads(fin.read())
                except FileNotFoundError:
                    logging.error(f"Annotation file not found: {annotation_file}")
                    continue

                for seq_name, seq_data in annotation.items():
                    if len(seq_data) < min_num_images:
                        continue
                    if seq_name in self.invalid_sequence:
                        continue
                    total_frame_num += len(seq_data)
                    self.data_store[seq_name] = seq_data

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Test"
        logging.info(f"{status}: Co3D Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Co3D Data length of training set: {len(self)}")

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
            filepath = anno["filepath"]
            # print('filepath',filepath)

            image_path = osp.join(self.CO3D_DIR, filepath)
            image = read_image_cv2(image_path)

            if self.load_depth:
                depth_path = image_path.replace("/images", "/depths") + ".geometric.png"
                depth_map = read_depth(depth_path, 1.0)

                if self.mask_depth:
                    mvs_mask_path = image_path.replace(
                        "/images", "/depth_masks"
                    ).replace(".jpg", ".png")
                    mvs_mask = cv2.imread(mvs_mask_path, cv2.IMREAD_GRAYSCALE) > 128
                    depth_map[~mvs_mask] = 0

                depth_map = threshold_depth_map(
                    depth_map, min_percentile=-1, max_percentile=98
                )
            else:
                depth_map = None
            # print('anno',anno)
            vggt_style = True
            if vggt_style:
                '''
                VGGT original pose conversion
                '''
                original_size = np.array(image.shape[:2])
                # print('before R',anno['R'])
                # print('T',anno['T'])
                # print('focal_length',anno['focal_length'])
                # print('principal_point',anno['principal_point'])
                extri_opencv, intri_opencv = convert_pt3d_to_opencv(anno, original_size)
                # print('after extri_opencv',extri_opencv)
                # print('after intri_opencv',intri_opencv)
            else:
                '''
                Adapted from pytorch3d
                '''
                original_size = np.array(image.shape[:2])
                h,w,_ = image.shape

                R = np.array(anno['R'])  # Rotation matrix (3x3)
                T = np.array(anno['T']).reshape(3, 1)  # Translation vector (3x1)
                focal_length = anno['focal_length']
                principal_point = anno['principal_point']
                print('before R',R)
                print('T',T)
                print('focal_length',focal_length)
                print('principal_point',principal_point)
                extri_opencv, intri_opencv = opencv_from_cameras_projection_numpy(R, T, principal_point, focal_length, h, w)
                print('after extri_opencv',extri_opencv)
                print('after intri_opencv',intri_opencv)




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
                filepath=filepath,
            )
            
            image = image/255.0
            image = image.astype(np.float32)

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
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
            "image_paths": image_paths,
        }
        return batch