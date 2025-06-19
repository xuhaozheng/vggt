import torch


'''
training utils for VGGT fine-tuning
'''


def normalize_camera_extrinsics_and_points_batch(
    extrinsics,
    cam_points=None,
    world_points=None,
    depths=None,
    scale_by_points=True,
    point_masks=None,
    mode="mean",
    seq_name=None,
):
    # Note this assumes we use cpu
    # extrinsics: (B, S, 3, 4)
    # world_points: (B, S, H, W, 3) or (*,3)
    # cam_points: same shape as world_points or something consistent
    # point_masks: (B, S, H, W) boolean mask if provided

    # check_valid_tensor(extrinsics, "extrinsics")
    # check_valid_tensor(cam_points, "cam_points")
    # check_valid_tensor(world_points, "world_points")
    # check_valid_tensor(depths, "depths")

    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    dtype = extrinsics.dtype


    # Convert extrinsics to homogeneous form: (B, N,4,4)
    if extrinsics.shape[-2:] == (3, 4):
        extrinsics_homog = torch.cat(
            [
                extrinsics,
                torch.zeros((B, S, 1, 4), device=device),
            ],
            dim=-2,
        )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = torch.linalg.inv(extrinsics_homog[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)

    # Force the first camera to be the identity
    # Do we really need this? Close it now
    # identity_4x4 = torch.eye(4, device=device, dtype=dtype)
    # new_extrinsics[:, 0] = identity_4x4


    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None


    if scale_by_points:
        new_cam_points = cam_points.clone()
        new_depths = depths.clone()

        dist = new_world_points.norm(dim=-1)
        dist_sum = (dist * point_masks).sum(dim=[1,2,3])
        valid_count = point_masks.sum(dim=[1,2,3])
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-3, max=1e3)


        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        if depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        if cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
        return new_extrinsics[:, :, :3], cam_points, new_world_points, depths
    else:
        return new_extrinsics[:, :, :3], cam_points, new_world_points, depths

def process_batch(batch, device):
    """
    Process a batch of data from the dataset.
    
    Args:
        batch (dict): A batch of data containing images, depths, extrinsics, intrinsics, point_masks, and world_points.
    
    Returns:
        dict: Processed batch with tensors for images, depths, extrinsics, intrinsics, point_masks, and world_points.
    """

    images = torch.stack(batch["images"]).to(device)
    depths = torch.stack(batch["depths"]).to(device)
    extrinsics = torch.stack(batch["extrinsics"]).to(device)
    intrinsics = torch.stack(batch["intrinsics"]).to(device)
    point_masks = torch.stack(batch["point_masks"]).to(device)
    world_points = torch.stack(batch["world_points"]).to(device)
    cam_points = torch.stack(batch["cam_points"]).to(device)
    '''
    images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
        B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
    query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
        Shape: [N, 2] or [B, N, 2], where N is the number of query points.
        Default: None

    Returns:
        dict: A dictionary containing the following predictions:
            - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
            - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
            - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
            - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
            - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
            - images (torch.Tensor): Original input images, preserved for visualization

            If query_points is provided, also includes:
            - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
            - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
            - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
    '''

    # print("images shape:", images.shape)
    # print("depths shape:", depths.shape)
    # print("extrinsics shape:", extrinsics.shape)
    # print("intrinsics shape:", intrinsics.shape)
    # print("point_masks shape:", point_masks.shape)
    # print("world_points shape:", world_points.shape)
    images = images.permute(1, 0, 4, 2, 3).contiguous()
    # depths = depths.permute(1, 0, 2, 3).unsqueeze(-1).contiguous()
    depths = depths.permute(1, 0, 2, 3).contiguous()
    point_masks = point_masks.permute(1, 0, 2, 3).contiguous()
    extrinsics = extrinsics.permute(1, 0, 2, 3).contiguous()
    intrinsics = intrinsics.permute(1, 0, 2, 3).contiguous()
    world_points = world_points.permute(1, 0, 2, 3, 4).contiguous()
    cam_points = cam_points.permute(1, 0, 2, 3, 4).contiguous()

    # print("depths shape:", depths.shape)
    # print("point_masks shape:", point_masks.shape)
    # print("intrinsics shape:", intrinsics.shape)
    # print("world_points shape:", world_points.shape)
    '''
    Normalize the point cloud data, translation and depth and check for NaN or Inf values.
    v1 original
    '''
    # extrinsics = convert_to_first_frame_coords(full_extrinsic_matrix(extrinsics))
    # print('after convert GT extrinsics',extrinsics)
    # gt_pts3d = world_points
    # valid_mask = point_masks
    # gt_pts3d = check_and_fix_inf_nan(gt_pts3d, "gt_pts3d")
    
    # gt_pts3d, gt_pts3d_scale = normalize_pointcloud(gt_pts3d, valid_mask)
    # extrinsics[:, :, :3, 3] = extrinsics[:, :, :3, 3]/gt_pts3d_scale # BxSx3
    '''
    v2 from jianyuan
    '''
    extrinsics[:, :, :3], cam_points, world_points, depths = normalize_camera_extrinsics_and_points_batch(extrinsics,cam_points,world_points,depths,point_masks=point_masks)


    processed_batch = {
        "images": images,
        "depths": depths,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "point_masks": point_masks,
        "world_points": world_points,
        "cam_points": cam_points,
    }
    return processed_batch


def find_translation_scale_ratios_per_pair(extrinsics_A: torch.Tensor, extrinsics_B: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Finds the scale ratio for each corresponding pair of extrinsic matrices.

    It assumes that extrinsics_A and extrinsics_B represent the same camera poses
    but with differently scaled translations. The scale ratio is computed as (scale_A / scale_B).

    Args:
        extrinsics_A (torch.Tensor): The first set of extrinsic matrices.
                                     Shape (..., 4, 4).
        extrinsics_B (torch.Tensor): The second set of extrinsic matrices.
                                     Shape (..., 4, 4).
        epsilon (float): A small value to prevent division by zero.

    Returns:
        torch.Tensor: A tensor containing the scale ratio for each pair.
                      Shape will be extrinsics_A.shape[:-2].
                      Returns 1.0 for pairs where the reference translation is near zero.
    """
    # Ensure shapes are compatible
    # if extrinsics_A.shape != extrinsics_B.shape:
    #     raise ValueError("Input tensors must have the same shape.")

    # Extract the translation vectors
    t_A = extrinsics_A[..., :3, 3]
    t_B = extrinsics_B[..., :3, 3]

    # Calculate the L2 norm (magnitude) of each translation vector
    mags_A = torch.linalg.norm(t_A, dim=-1)
    mags_B = torch.linalg.norm(t_B, dim=-1)

    # Initialize the output ratios with a default value of 1.0.
    # This handles cases where the denominator is zero, assuming no scale change.
    ratios = torch.ones_like(mags_A)

    # Create a mask to identify pairs where the reference magnitude is large enough for a stable division.
    valid_mask = mags_B > epsilon

    # Calculate the ratio of magnitudes only for the valid pairs
    # and update the ratios tensor at the valid locations.
    ratios[valid_mask] = mags_A[valid_mask] / mags_B[valid_mask]

    return ratios