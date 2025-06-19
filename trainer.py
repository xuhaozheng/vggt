# train.py

import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import from your new modules
from training.config import Config
from training.data.datasets.co3d import Co3dDataset
from training.data.datasets.dynamic_replica import DynamicReplicaDataset
from vggt.models.vggt import VGGT
from training.data.dynamic_dataloader import build_dynamic_dataloader
from training.train_utils import process_batch, find_translation_scale_ratios_per_pair

from training.loss import camera_loss, depth_loss, point_loss
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
import torch.distributed as dist

torch.set_printoptions(sci_mode=False)

def setup_distributed(config):
    os.environ['MASTER_ADDR'] = config.MASTER_ADDR
    os.environ['MASTER_PORT'] = config.MASTER_PORT
    os.environ['WORLD_SIZE'] = str(config.WORLD_SIZE)
    os.environ['RANK'] = str(config.RANK)
    dist.init_process_group(backend="nccl", init_method="env://")


def main():
    # 1. SETUP
    config = Config()
    setup_distributed(config) # Uncomment for distributed training
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


    # 2. DATA
    if config.DS_TYPE == 'CO3D':
        train_dataset = Co3dDataset(
            common_conf=config.common_conf,
            split="train",
            CO3D_DIR=config.CO3D_DIR,
            CO3D_ANNOTATION_DIR=config.CO3D_ANNOTATION_DIR,
        )
        val_dataset = Co3dDataset(
            common_conf=config.common_conf,
            split="test",
            CO3D_DIR=config.CO3D_DIR,
            CO3D_ANNOTATION_DIR=config.CO3D_ANNOTATION_DIR,
        )
    elif config.DS_TYPE == 'DR':
        train_dataset = DynamicReplicaDataset(
            common_conf=config.common_conf,
            split="train",
            DR_DIR=config.DR_DIR,
            DR_ANNOTATION_DIR=config.DR_ANNOTATION_DIR,
        )
        val_dataset = DynamicReplicaDataset(
            common_conf=config.common_conf,
            split="test",
            DR_DIR=config.DR_DIR,
            DR_ANNOTATION_DIR=config.DR_ANNOTATION_DIR,
        )
    # Build dynamic dataloaders
    train_loader = build_dynamic_dataloader(
        dataset=train_dataset,
        common_config=config.common_conf,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
        persistent_workers=False,
        seed=42,
        epoch=0,
        max_img_per_gpu=config.max_img_per_gpu,
    )

    val_loader = build_dynamic_dataloader(
        dataset=val_dataset,
        common_config=config.common_conf,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=None,
        persistent_workers=False,
        seed=42,
        epoch=0,
        max_img_per_gpu=config.max_img_per_gpu,
    )
    # 3. MODEL, OPTIMIZER
    model = VGGT.from_pretrained(config.MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,weight_decay = 0.05)

    # 4. TRAINING LOOP
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0  # Initialize cumulative training loss
        for batch_idx, batch in enumerate(train_loader):
            # Move batch data to the device
            batch = process_batch(batch, device)

            images = batch["images"]

            gt_extrinsics = batch["extrinsics"]

            # Forward pass
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)

            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
            # print('pred extrinsic',extrinsic)

            ratios = find_translation_scale_ratios_per_pair(gt_extrinsics, extrinsic)
            print("ratios:", ratios)
            # exit()

            # Compute losses
            camera_loss_dict, _ = camera_loss(
                predictions["pose_enc"], batch, loss_type="l1"
            )
            depth_loss_dict = depth_loss(
                predictions["depth"], predictions["depth_conf"], batch
            )
            point_loss_dict = point_loss(
                predictions["world_points"], predictions["world_points_conf"], batch
            )

            # Compute total_loss for the current batch
            total_loss = 0
            # total_loss = camera_loss_dict['loss_camera'] + depth_loss_dict['loss_depth'] + point_loss_dict['loss_point']
            for loss_dict in [camera_loss_dict, depth_loss_dict, point_loss_dict]:
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:

                        total_loss += value  # Accumulate the loss
                        # Log individual loss components to TensorBoard
                        writer.add_scalar(f"Train/{key}", value.item(), epoch * len(train_loader) + batch_idx)
                        # Print the loss to the console
                        print(f"Epoch {epoch}, Batch {batch_idx}, {key}: {value.item()}")

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            # Accumulate total_loss for the epoch
            epoch_train_loss += total_loss.item()

        # Log average training loss for the epoch
        average_train_loss = epoch_train_loss / len(train_loader)
        writer.add_scalar("Train/Average_Loss", average_train_loss, epoch)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Compute validation losses
                camera_loss_dict, _ = camera_loss(
                    predictions["pose_enc"], batch, loss_type="l1"
                )
                depth_loss_dict = depth_loss(
                    predictions["depth"], predictions["depth_conf"], batch
                )
                point_loss_dict = point_loss(
                    predictions["world_points"], predictions["world_points_conf"], batch
                )

                # Combine all validation losses
                batch_val_loss = 0
                for loss_dict in [camera_loss_dict, depth_loss_dict, point_loss_dict]:
                    for key, value in loss_dict.items():
                        if isinstance(value, torch.Tensor) and value.numel() == 1:
                            batch_val_loss += value
                            # Log individual validation loss components to TensorBoard
                            writer.add_scalar(f"Validation/{key}", value.item(), epoch * len(val_loader) + batch_idx)

                            # Print the loss to the console
                            print(f"Epoch {epoch}, Validation Batch {batch_idx}, {key}: {value.item()}")

                total_val_loss += batch_val_loss

        # Log total losses after each epoch
        writer.add_scalar("Train/Total_Loss", total_loss.item(), epoch)
        writer.add_scalar("Validation/Total_Loss", total_val_loss, epoch)

        print(
            f"Epoch {epoch + 1}/{config.NUM_EPOCHS}, Training Loss: {total_loss.item()}"
        )

    writer.close()

if __name__ == "__main__":
    main()