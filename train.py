# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# train with a single gpu
python train.py

# train with 8 gpus
torchrun --standalone --nproc_per_node=8 train.py

# train with multi-node multi-gpu, run
sbatch sbatch_run.sh
"""

import math
import os
import random
import shutil
import time
from datetime import datetime

import hydra
import omegaconf

import torch
import torch.distributed as dist
import tqdm
import webdataset as wds
import yaml
from efm3d.aria.tensor_wrapper import custom_collate_fn
from efm3d.dataset.augmentation import ColorJitter, PointDropSimple, PointJitter
from efm3d.dataset.efm_model_adaptor import load_atek_wds_dataset_as_efm_train
from efm3d.dataset.vrs_dataset import preprocess
from efm3d.dataset.wds_dataset import get_tar_sample_num
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter


DATA_PATH = "./data/ase_train"
MAX_LR = 2e-4
MIN_LR = MAX_LR * 0.1
BATCH_SIZE = 2
MAX_EPOCHS = 40
MAX_SAMPLES_PER_EPOCH = 100000
SAVE_EVERY_EPOCHS = 5  # save the model every
LOG_STEP = 5  # print error every


def get_lr(it, warmup_its, max_its, max_lr, min_lr):
    """
    cosine learning rate scheduler, `it` can be either step or epoch.
    """
    # learning rate scheduler
    # linear warmup for warmup_epochs
    if it < warmup_its:
        return max_lr * (it + 1) / warmup_its

    # return min_lr if epoch > max_epochs
    if it > max_its:
        return min_lr

    # cosine annealing
    decay_ratio = (it - warmup_its) / (max_its - warmup_its)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1.0 -> 0.0
    return min_lr + coeff * (max_lr - min_lr)


def get_dataloader(
    data_path,
    batch_size,
    world_size,
    max_samples_per_epoch,
    epoch_sample_ratio=1.0,
    tar_yaml="train_tars.yaml",
):
    assert (
        epoch_sample_ratio > 0 and epoch_sample_ratio <= 1.0
    ), f"{epoch_sample_ratio} is the ratio ([0, 1]) of samples used in each epoch"

    tar_yaml = os.path.join(data_path, tar_yaml)
    with open(tar_yaml, "r") as f:
        tar_list = yaml.safe_load(f)["tars"]
    tar_list = [os.path.join(data_path, tar_name) for tar_name in tar_list]

    # check existence
    for tar in tar_list:
        assert os.path.exists(tar), f"{tar} not exists"
    random.shuffle(tar_list)
    dataset = load_atek_wds_dataset_as_efm_train(
        urls=tar_list,
        atek_to_efm_taxonomy_mapping_file=f"{os.path.dirname(__file__)}/efm3d/config/taxonomy/atek_to_efm.csv",
        batch_size=batch_size,
        collation_fn=custom_collate_fn,
    )

    samples_per_tar = get_tar_sample_num(tar_list[0])
    dataset_size = len(tar_list) * samples_per_tar
    dataset_size = min(dataset_size, max_samples_per_epoch)
    dataset_size = int(dataset_size * epoch_sample_ratio)

    batches_per_epoch = int(dataset_size // (batch_size * world_size))
    dataloader = wds.WebLoader(
        dataset,
        num_workers=batch_size,
        pin_memory=True,
        prefetch_factor=2,
        batch_size=None,
        shuffle=False,
    )
    dataloader = dataloader.with_epoch(batches_per_epoch)
    dataloader = dataloader.with_length(batches_per_epoch)

    return dataloader


ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group("nccl")
    DDP_RANK = int(os.environ["RANK"])
    DDP_LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    DDP_WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{DDP_LOCAL_RANK}"
    print(f"==> setting device to {device}")
    torch.cuda.set_device(device)
    master_process = DDP_RANK == 0
else:
    DDP_RANK = 0
    DDP_LOCAL_RANK = 0
    DDP_WORLD_SIZE = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_config = omegaconf.OmegaConf.load("efm3d/config/evl_train.yaml")
model = hydra.utils.instantiate(model_config)
model = model
model.to(device)
if ddp:
    model = DDP(model, device_ids=[DDP_LOCAL_RANK])
raw_model = model.module if ddp else model

train_dataloader = get_dataloader(
    DATA_PATH,
    BATCH_SIZE,
    DDP_WORLD_SIZE,
    max_samples_per_epoch=MAX_SAMPLES_PER_EPOCH,
    tar_yaml="train_tars.yaml",
)
val_dataloader = get_dataloader(
    DATA_PATH,
    BATCH_SIZE,
    DDP_WORLD_SIZE,
    max_samples_per_epoch=MAX_SAMPLES_PER_EPOCH,
    tar_yaml="val_tars.yaml",
)
optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR)

if master_process:
    exp_name = f"efm3d_train_b{BATCH_SIZE}g{DDP_WORLD_SIZE}e{MAX_EPOCHS}lr{str(MAX_LR)}_{datetime.fromtimestamp(time.time()).strftime('%y-%m-%d-%H-%M-%S')}"
    log_dir = os.path.join("tb_logs", exp_name)
    writer = SummaryWriter(log_dir=log_dir)

color_jitter = ColorJitter(
    brightness=0.5,
    contrast=0.3,
    saturation=0.3,
    hue=0.05,
    sharpness=2.0,
    snippet_jitter=True,
)
point_drop = PointDropSimple(max_dropout_rate=0.8)
point_jitter = PointJitter(depth_std_scale_min=1.0, depth_std_scale_max=6.0)
augmentations = [color_jitter, point_drop, point_jitter]

step = 0
val_step = 0
# main loop
for epoch in range(MAX_EPOCHS):
    # train
    model.train()
    for batch in tqdm.tqdm(train_dataloader):
        start = time.time()
        optimizer.zero_grad()

        batch = preprocess(batch, device, aug_funcs=augmentations)
        output = model(batch)
        losses, total_loss = raw_model.compute_losses(output, batch)

        total_loss.backward()

        # epoch-based lr scheduler
        lr = get_lr(
            epoch, warmup_its=5, max_its=MAX_EPOCHS, max_lr=MAX_LR, min_lr=MIN_LR
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if ddp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
        max_norm = 1.0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        time_per_it = time.time() - start

        if master_process and step % LOG_STEP == 0:
            print(
                f"E:s-{epoch}:{step} | loss {total_loss.item():.03f} | lr {lr:.06f} | norm {norm} | time {time_per_it:.02f}s/it"
            )

            # log training
            writer.add_scalar("train/loss", total_loss.item(), step)
            for stream in losses:
                for loss_name in losses[stream]:
                    writer.add_scalar(
                        f"train/loss/{stream}/{loss_name}",
                        losses[stream][loss_name].item(),
                        step,
                    )
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/iter_sec", time_per_it, step)

            # log images (log every `10xlog_step` since writing video is slow)
            if step % (10 * LOG_STEP) == 0:
                imgs = raw_model.log_single(batch, output, batch_idx=0)
                for k, v in imgs.items():
                    vid = torch.tensor(v.transpose((0, 3, 1, 2))).unsqueeze(0)
                    writer.add_video(f"train/{k}", vid, global_step=step, fps=10)
        step += 1

    # val
    model.eval()
    for batch in tqdm.tqdm(val_dataloader):
        with torch.no_grad():
            start = time.time()
            batch = preprocess(batch, device, aug_funcs=augmentations)
            output = model(batch)
            losses, total_loss = raw_model.compute_losses(output, batch)
            if ddp:
                dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
            time_per_it = time.time() - start

        if master_process and val_step % LOG_STEP == 0:
            print(
                f"E:s-{epoch}:{val_step} | loss {total_loss.item():.03f} | time {time_per_it:.02f}s/it"
            )

            # log val
            if val_step % LOG_STEP == 0:
                writer.add_scalar("val/loss", total_loss.item(), val_step)
                for stream in losses:
                    for loss_name in losses[stream]:
                        writer.add_scalar(
                            f"val/loss/{stream}/{loss_name}",
                            losses[stream][loss_name].item(),
                            val_step,
                        )
                writer.add_scalar("val/iter_sec", time_per_it, val_step)

            # log images
            if val_step % (10 * LOG_STEP) == 0:
                imgs = raw_model.log_single(batch, output, batch_idx=0)
                for k, v in imgs.items():
                    vid = torch.tensor(v.transpose((0, 3, 1, 2))).unsqueeze(0)
                    writer.add_video(f"val/{k}", vid, global_step=val_step, fps=10)
        val_step += 1

    # save model
    if master_process and (epoch + 1) % SAVE_EVERY_EPOCHS == 0:
        ckpt_path = os.path.join(
            log_dir, f"model_e{epoch}s{step}_l{total_loss.item():.02f}.pth"
        )
        last_ckpt_path = os.path.join(log_dir, "last.pth")
        torch.save(
            {"state_dict": raw_model.state_dict(), "optimizer": optimizer.state_dict()},
            ckpt_path,
        )
        shutil.copy(ckpt_path, last_ckpt_path)

if master_process:
    writer.close()
if ddp:
    destroy_process_group()
