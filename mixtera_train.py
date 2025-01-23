import copy
import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime
from functools import partial

import numpy as np
from omegaconf import OmegaConf
import torch
import wandb
from torch import optim

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from open_clip_train.data import get_data
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.logger import setup_logging
from open_clip_train.main import random_seed, natural_key, get_latest_checkpoint, LATEST_CHECKPOINT_NAME
from open_clip_train.params import parse_args
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from open_clip_train.train import train_one_epoch, evaluate
from open_clip_train.file_utils import pt_load, check_exists, start_sync_process, remote_sync

def main(args):
    assert torch.cuda.is_available(), "Cuda not available."

    args = parse_args(args)

    config = OmegaConf.load("./config.yml")

    device = init_distributed_device(args)

    args.name = config.setup.name
    args.wandb = config.setup.wandb
    args.distributed = True
    args.use_bn_sync = config.training.use_bn_sync
    args.seed = config.setup.seed
    args.siglip = config.training.siglip
    args.resume = config.training.resume
    args.grad_checkpointing = config.training.grad_checkpointing
    args.force_quick_gelu = config.training.force_quick_gelu
    args.force_custom_text = config.training.force_custom_text
    args.model = config.training.model
    args.force_patch_dropout = config.training.force_patch_dropout
    args.force_image_size = config.training.force_image_size
    args.cache_dir = config.setup.cache_dir
    args.use_bnb_linear = config.training.use_bnb_linear
    args.opt = config.training.opt
    args.ddp_static_graph = config.training.ddp_static_graph
    resume_latest = args.resume == 'latest'
    args.lr_scheduler = config.training.lr_scheduler
    args.acum_freq = config.training.acum_freq
    args.wandb_notes = config.setup.wandb_notes
    args.delete_previous_checkpoint = config.training.delete_previous_checkpoint
    args.train_data = config.training.train_data
    args.train_num_samples = config.training.train_num_samples
    args.dataset_type = config.training.dataset_type
    args.distill = False

    args.save_most_recent = config.training.save_most_recent
    args.torchcompile = config.training.torchcompile
    args.logs = config.setup.logs
    args.log_local = config.setup.log_local
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")

    args.log_level = logging.INFO
    setup_logging(args.log_path, args.log_level)

    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
    
    if resume_latest:
        raise NotImplementedError("Resuming from latest checkpoint is not implemented yet.")

    logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')

    random_seed(args.seed, 0)
    model_kwargs = {}

    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10
    
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=None,
        image_std=None,
        image_interpolation=None,
        image_resize_mode=None,  # only effective for inference
        aug_cfg={},
        pretrained_image=False,
        output_dict=True,
        cache_dir=args.cache_dir,
        **model_kwargs,
    )

    random_seed(args.seed, args.rank)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()
    
    if is_master(args):
        logging.info("Model training with Mixtera is starting.")
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")
    
    if args.use_bn_sync:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], static_graph=args.ddp_static_graph)

    optimizer = None
    scaler = None

    opt = args.opt 

    # If some params are not passed, we use the default values based on model name.
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    if opt == 'adamw':
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt}")

    if is_master(args):
        defaults = copy.deepcopy(optimizer.defaults)
        defaults['weight_decay'] = args.wd
        defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
        logging.info(
            f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
        )
    
    if args.precision == "amp":
        try:
            scaler = torch.amp.GradScaler(device=device)
        except (AttributeError, TypeError) as e:
            scaler = torch.cuda.amp.GradScaler()
    
    start_epoch = 0 
    if args.resume:
        raise NotImplementedError("Resuming from a checkpoint is not implemented yet.")
    
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'

    scheduler = None
    total_steps = 1000 # (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs

    if args.lr_scheduler == "cosine":
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    elif args.lr_scheduler == "const":
        scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
    else:
        raise ValueError(f"Unknown lr scheduler: {args.lr_scheduler}")
    
    args.save_logs = is_master(args)

    if args.wandb and is_master(args):
        logging.debug('Starting wandb.')
        #args.train_sz = data["train"].dataloader.num_samples
        #if args.val_data is not None:
        #    args.val_sz = data["val"].dataloader.num_samples

        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')

        if args.grad_checkpointing and args.distributed:
            logging.info('Disabling DDP dynamo optimizer when grad checkpointing enabled.')
            # As of now (~PyTorch 2.4/2.5), compile + grad checkpointing work, but DDP optimizer must be disabled
            torch._dynamo.config.optimize_ddp = False

        model = torch.compile(original_model)
    
    loss = create_loss(args)
    
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, None, args, tb_writer=None)
        completed_epoch = epoch + 1

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    if args.wandb and is_master(args):
        wandb.finish()

if __name__ == "__main__":
    main(sys.argv[1:])

    

    









    

