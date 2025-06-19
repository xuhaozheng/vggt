
import os


class CommonConfig:
    debug = False
    training = True
    get_nearby = False
    load_depth = True
    inside_random = True
    # - img_size: Default is 518
    # - patch_size: Default is 14
    # - augs.scales: Default is [0.8, 1.2]
    # - rescale: Default is True
    # - rescale_aug: Default is True
    # - landscape_check: Default is True
    img_size = 518
    patch_size = 14
    rescale = True
    rescale_aug = True
    landscape_check = True
    aspects = [0.5, 1.0]  # Aspect ratio range need to customize!!
    img_nums = [2, 12]  # Image number range need to customize!!


class Config:
    # --- PATHS ---
    ROOT_DIR = "/media/neurodragon/Extreme SSD1"
    DR_DIR = os.path.join(ROOT_DIR, 'dynamic_replica/dynamic_replica')
    DR_ANNOTATION_DIR = os.path.join(ROOT_DIR, 'dynamic_replica/dynamic_replica')
    CO3D_DIR = os.path.join(ROOT_DIR, "co3d_data")
    CO3D_ANNOTATION_DIR = os.path.join(ROOT_DIR, "co3d_process")
    LOG_DIR = "runs/experiment_1" # For TensorBoard

    # --- DATASET ---
    DS_TYPE = 'CO3D'  # 'CO3D' or 'DR'
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    max_img_per_gpu = 24  # Maximum number of images per GPU, can be customized

    # --- TRAINING ---
    DEVICE = "cuda"
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50

    # --- MODEL ---
    MODEL_NAME = "facebook/VGGT-1B"

    # --- DISTRIBUTED TRAINING ---
    MASTER_ADDR = 'localhost'
    MASTER_PORT = '12355'
    WORLD_SIZE = 1
    RANK = 0

    common_conf = CommonConfig()