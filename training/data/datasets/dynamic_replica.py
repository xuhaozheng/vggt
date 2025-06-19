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
from PIL import Image
# from vggt.training.data.dataset_util import *
# from vggt.training.data.base_dataset import BaseDataset

from training.data.base_dataset import BaseDataset
from training.data.dataset_util import *
import torch
from numpy.linalg import inv
### From https://github.com/facebookresearch/dynamic_stereo/blob/main/notebooks/Dynamic_Replica_demo.ipynb

def flowreader(flow_path):
    with Image.open(flow_path) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        flow = np.frombuffer(
            np.array(depth_pil, dtype=np.uint16), dtype=np.float16
        ).astype(np.float32).reshape((depth_pil.size[1], depth_pil.size[0]))
    flow_res = np.stack([flow[:,:flow.shape[1]//2], flow[:,flow.shape[1]//2:]],axis=-1)
    return flow_res

def convert_ndc_to_pixel_intrinsics(
    focal_length_ndc, principal_point_ndc, image_width, image_height, intrinsics_format='ndc_isotropic'
):
    f_x_ndc, f_y_ndc = focal_length_ndc
    c_x_ndc, c_y_ndc = principal_point_ndc

    # Compute half image size
    half_image_size_wh_orig = np.array([image_width, image_height]) / 2.0

    # Determine rescale factor based on intrinsics_format
    if intrinsics_format.lower() == "ndc_norm_image_bounds":
        rescale = half_image_size_wh_orig  # [image_width/2, image_height/2]
    elif intrinsics_format.lower() == "ndc_isotropic":
        rescale = np.min(half_image_size_wh_orig)  # scalar value
    else:
        raise ValueError(f"Unknown intrinsics format: {intrinsics_format}")

    # Convert focal length from NDC to pixel coordinates
    if intrinsics_format.lower() == "ndc_norm_image_bounds":
        focal_length_px = np.array([f_x_ndc, f_y_ndc]) * rescale
    elif intrinsics_format.lower() == "ndc_isotropic":
        focal_length_px = np.array([f_x_ndc, f_y_ndc]) * rescale

    # Convert principal point from NDC to pixel coordinates
    principal_point_px = half_image_size_wh_orig - np.array([c_x_ndc, c_y_ndc]) * rescale

    # Construct the intrinsics matrix in pixel coordinates
    K_pixel = np.array([
        [focal_length_px[0], 0,                principal_point_px[0]],
        [0,                 focal_length_px[1], principal_point_px[1]],
        [0,                 0,                 1]
    ])

    return K_pixel

def load_16big_png_depth(depth_png):
        with Image.open(depth_png) as depth_pil:
            # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
            # we cast it to uint16, then reinterpret as float16, then cast to float32
            depth = (
                np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depth_pil.size[1], depth_pil.size[0]))
            )
        return depth

class DynamicReplicaDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        DR_DIR: str = None,
        DR_ANNOTATION_DIR: str = None,
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

        if DR_DIR is None:
            raise ValueError("Both PD_DIR and PD_ANNOTATION_DIR must be specified.")
        '''
        Defalt values from pointodysseyDUST3R
        '''
        self.dataset_label = 'dynamic_replica'
        self.S = 2 # stride
        self.verbose = True

        if split == "train":
            split_name_list = ["train"]
            self.len_train = len_train
        elif split == "test":
            split_name_list = ["test"]
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        ### Manual defined dataset parameters
        quick = False
        self.dataset_location = DR_DIR
        dset = split
        self.flow_forward = "none"
        self.duplicate_img = False
        self.use_trajectory = False
        
        ### Data Variables
        self.data_store = {}
        # self.depth_paths = []
        # self.normal_paths = []
        # self.traj_paths = []
        # self.annotation_paths = []
        # self.full_idxs = []
        # self.sample_stride = []

        # self.subdirs = []
        # self.sequences = []
        # self.subdirs.append(os.path.join(dataset_location))

        anno_path = os.path.join(self.dataset_location, 'frame_annotations_train.json')
        with open(anno_path, 'r') as f:
            self.anno = json.load(f)
        if quick:
            self.anno = self.anno[:1000]
        # print(len(self.anno))   # 289800 entries
        # print(self.anno[0])

        #organize anno by 'sequence_name'
        # anno_by_seq = {}
        for annotation in self.anno:
            sequence_name = annotation['sequence_name']
            image_path = annotation['image']['path']
            main_folder = image_path.split('/')[0]  # Get the first part before first '/'
            
            # Skip this annotation if the main folder ends with "right"
            if main_folder.endswith("_right"):
                continue
            if sequence_name not in self.data_store:
                self.data_store[sequence_name] = []

            # annotation['image']['path'] = os.path.join(
            #     dataset_location, annotation['image']['path']
            # )
            # annotation['depth']['path'] = os.path.join(
            #     dataset_location, annotation['depth']['path']
            # )
            
            self.data_store[sequence_name].append(annotation)    

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        # print("All sequence names:", self.sequence_list)

        if quick:
            first_sequence_name = self.sequence_list[0]
            print("First sequence name:", first_sequence_name)

            # 3. Shorten self.data_store to only contain the first sequence
            self.data_store = {first_sequence_name: self.data_store[first_sequence_name]}

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

        # Add optical flow processing
        flows = []
        flow_masks = []
        trajectories = []
        processed_tracks = []
        
        for anno in annos:   
            impath = os.path.join(self.dataset_location,anno['image']['path'])
            depthpath = os.path.join(self.dataset_location,anno['depth']['path'])

            rgb_image = imread_cv2(impath)
            depthmap = load_16big_png_depth(depthpath)

            ### Optical flow processing
            if self.flow_forward == "foward":
                # print(anno['flow_forward']['path'])
                flow_path = os.path.join(self.dataset_location,anno['flow_forward']['path'])
                flow_mask_path = os.path.join(self.dataset_location,anno['flow_forward_mask']['path'])
            elif self.flow_forward == "backward":  
                flow_path = os.path.join(self.dataset_location,anno['flow_backward']['path'])
                flow_mask_path = os.path.join(self.dataset_location,anno['flow_backward_mask']['path'])
            else:
                flow_path = None
                flow_mask_path = None
            
            ### Trajectory processing
            traj_path = os.path.join(self.dataset_location,anno['trajectories']['path'])
            traj = torch.load(traj_path)

            # flow = flowreader(flow_path)
            # flow_mask = np.array(Image.open(flow_mask_path))

            viewpoint = anno['viewpoint']
            # load camera params
            R = np.array(viewpoint['R']).astype(np.float32) ### extrinsic matrix
            t = np.array(viewpoint['T']).astype(np.float32) ### translation vector
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3,:3] = R.T
            camera_pose[:3,3] = -R.T @ t

            focals = np.array(viewpoint['focal_length']).astype(np.float32) 
            pp = np.array(viewpoint['principal_point']).astype(np.float32) 
            intrinsics_format = viewpoint['intrinsics_format']

            intrinsic = convert_ndc_to_pixel_intrinsics(focals, pp, rgb_image.shape[1], rgb_image.shape[0],
                                                         intrinsics_format=intrinsics_format)
            intrinsic = intrinsic.astype(np.float32)

            # extri_opencv = inv(camera_pose)
            extri_opencv = camera_pose
            intri_opencv = intrinsic
            image = rgb_image
            depth_map = depthmap

            if self.use_trajectory:
                track = traj["traj_2d"][:,:2]
            else:
                track = None

            original_size = np.array(image.shape[:2])

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                track,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                track = track,
                filepath=impath,
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
            image_paths.append(impath)
            original_sizes.append(original_size)
            ### new added branch
            # flows.append(flow)
            # flow_masks.append(flow_mask)
            trajectories.append(traj)
            processed_tracks.append(track)

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
            # "flows": flows,
            # "flow_masks": flow_masks,
            "image_paths": image_paths,
        }
        if self.use_trajectory:
            batch["trajectories"] = trajectories
            batch["processed_tracks"] = processed_tracks
    
        return batch