from training.data.datasets.co3d_test import Co3dDataset
from training.data.datasets.pointodyssey import PointOdysseyDataset
import numpy as np
import matplotlib.pyplot as plt

# Configuration for the dataset
class CommonConfig:
    debug = False
    training = False
    get_nearby = False
    load_depth = True
    inside_random = True
    img_size = 518
    patch_size = 14
    rescale = True
    rescale_aug = True
    landscape_check = True

common_conf = CommonConfig()

def reproject_and_visualize(batch, index=0):
    """Reproject 3D points to the 2D image plane and visualize."""
    image = batch["images"][index]
    extrinsics = batch["extrinsics"][index]  # 3x4 matrix
    intrinsics = batch["intrinsics"][index]  # 3x3 matrix
    world_points = batch["world_points"][index]  # HxWx3 array
    print('world_points shape:', world_points.shape)

    # Flatten the world points to (N, 3) for projection
    H, W, _ = world_points.shape
    world_points_flat = world_points.reshape(-1, 3)

    # Convert world points to homogeneous coordinates
    world_points_h = np.hstack((world_points_flat, np.ones((world_points_flat.shape[0], 1))))

    # Project world points to camera coordinates
    cam_points = extrinsics @ world_points_h.T  # Shape: (3, N)

    # Filter valid points (e.g., points in front of the camera)
    valid_mask = cam_points[2, :] > 0  # Z > 0
    cam_points = cam_points[:, valid_mask]

    # Project camera points to the image plane
    image_points_h = intrinsics @ cam_points  # Shape: (3, N)
    image_points = image_points_h[:2, :] / image_points_h[2, :]  # Normalize by Z

    # Visualize the image and reprojected points
    plt.figure(figsize=(10, 5))
    plt.imshow(image)
    plt.scatter(image_points[0, :], image_points[1, :], c='r', s=5, label="Reprojected Points")
    plt.title("Reprojected 3D Points on Image")
    plt.legend()
    plt.axis("off")
    plt.show()
def reproject_and_visualize2(batch, index=0):
    """Reproject 3D points to the 2D image plane and visualize."""
    image = batch["images"][index]
    extrinsics = batch["extrinsics"][index][:3,:]  # 3x4 matrix (OpenCV-style)
    intrinsics = batch["intrinsics"][index]  # 3x3 matrix (OpenCV-style)
    world_points = batch["world_points"][index]  # HxWx3 array

    # Flatten the world points to (N, 3) for projection
    H, W, _ = world_points.shape
    world_points_flat = world_points.reshape(-1, 3)

    # Convert world points to homogeneous coordinates
    world_points_h = np.hstack((world_points_flat, np.ones((world_points_flat.shape[0], 1))))

    # Project world points to the image plane
    image_points_h = intrinsics @ (extrinsics @ world_points_h.T)  # Shape: (3, N)
    image_points = image_points_h[:2, :] / image_points_h[2, :]  # Normalize by Z

    # Visualize the image and reprojected points
    plt.figure(figsize=(10, 5))
    plt.imshow(image)
    plt.scatter(image_points[0, :], image_points[1, :], c='r', s=5, label="Reprojected Points")
    plt.title("Reprojected 3D Points on Image")
    plt.legend()
    plt.axis("off")
    plt.show()

def visualize_depth_and_3d(batch, index=0):
    """Visualize the depth map and reconstructed 3D points."""
    depth = batch["depths"][index]
    cam_points = batch["cam_points"][index]  # HxWx3 array

    # Flatten cam_points for visualization
    H, W, _ = cam_points.shape
    cam_points_flat = cam_points.reshape(-1, 3)

    # Visualize the depth map
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(depth, cmap="viridis")
    plt.title("Depth Map")
    plt.axis("off")

    # Visualize the 3D points
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(cam_points_flat[:, 0], cam_points_flat[:, 1], cam_points_flat[:, 2], s=1, c=cam_points_flat[:, 2], cmap="viridis")
    ax.set_title("3D Points")
    plt.show()

CO3D_DIR = "/media/deeplearner/d9f92773-3994-4f26-bb20-deb1d94e0236/haozheng/co3d_data"
CO3D_ANNOTATION_DIR = "/media/deeplearner/d9f92773-3994-4f26-bb20-deb1d94e0236/haozheng/co3d_process"
# CO3D_ANNOTATION_DIR = "/media/deeplearner/d9f92773-3994-4f26-bb20-deb1d94e0236/haozheng/co3d_data"
PD_DIR = '/media/deeplearner/Extreme SSD/point_odyssey'
PD_ANNOTATION_DIR = '/media/deeplearner/Extreme SSD/point_odyssey'

DS_TYPE = 'pointodyssey'  # Change to 'co3d' or 'pointodyssey' as needed
# Initialize the dataset
if DS_TYPE == 'co3d':
    dataset = Co3dDataset(
        common_conf=common_conf,
        split="train",
        CO3D_DIR=CO3D_DIR,
        CO3D_ANNOTATION_DIR=CO3D_ANNOTATION_DIR,
    )
elif DS_TYPE == 'pointodyssey':
    dataset = PointOdysseyDataset(
        common_conf=common_conf,
        split="train",
        PD_DIR=PD_DIR,
        PD_ANNOTATION_DIR=PD_ANNOTATION_DIR,
    )

# Access a specific sample using an index
index = 0  # Replace with the index you want to test
sample = dataset.get_data(seq_index=index, img_per_seq=1,ids=[1])

# Print or visualize the sample
print("Sample keys:", sample.keys())
print("Sequence name:", sample["seq_name"])
print("Number of images:", len(sample["images"]))

# Visualize the first image and depth map
image = sample["images"][0]
depth = sample["depths"][0]

print("Visualizing Image and Depth Map...")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Image")
plt.axis("off")

if depth is not None:
    plt.subplot(1, 2, 2)
    plt.imshow(depth, cmap="viridis")
    plt.title("Depth Map")
    plt.axis("off")

plt.show()

# Reproject and visualize 3D points
print("Reprojecting and Visualizing 3D Points...")
# reproject_and_visualize(sample)
reproject_and_visualize2(sample)

# Visualize depth and reconstructed 3D points
print("Visualizing Depth and Reconstructed 3D Points...")
# visualize_depth_and_3d(sample)