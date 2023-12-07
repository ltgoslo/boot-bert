# coding=utf-8

import os
import os.path
import argparse
from tqdm import tqdm
from itertools import count
from socket import gethostname

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tokenizers import Tokenizer
from lamb import Lamb
from config import BertConfig

from model import Bert

from utils import cosine_schedule_with_warmup, is_main_process, get_rank, seed_everything, get_world_size, CosineWDSchedule
from dataset import Dataset, Collate


if int(os.environ["SLURM_PROCID"]) == 0:
    import wandb


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_path", default="../data/processed/cached_{sequence_length}.txt.gz", type=str, help="The input data dir. Should contain .hdf5 files for the task.")
    parser.add_argument("--config_file", default="../configs/base.json", type=str, help="The BERT model config")
    parser.add_argument("--output_dir", default="../checkpoints/mae_byol_MLM_only", type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--vocab_path", default="../tokenizer.json", type=str, help="The vocabulary the BERT model will train on.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to a previous checkpointed training state.")

    # Other parameters
    parser.add_argument("--optimizer", default="lamb", type=str)
    parser.add_argument("--scheduler", default="cosine", type=str)
    parser.add_argument("--seq_length", default=256, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size for training per GPUs and per grad accumulation step.")
    parser.add_argument("--learning_rate", default=0.5e-2, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps", default=31250 * 2, type=int, help="Total number of training steps to perform.")
    parser.add_argument("--long_after", default=1.0, type=float)
    parser.add_argument("--ema_decay_start", default=0.996, type=float)
    parser.add_argument("--ema_decay_end", default=1.0, type=float)
    parser.add_argument("--mask_p_start", default=0.15, type=float)
    parser.add_argument("--mask_p_end", default=0.15, type=float)
    parser.add_argument("--mlm_weight_start", default=0.1, type=float)
    parser.add_argument("--mlm_weight_end", default=0.1, type=float)
    parser.add_argument("--label_smoothing", default=0.0, type=float) 
    parser.add_argument("--warmup_proportion", default=0.016, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--log_freq', type=int, default=10, help='frequency of logging loss.')
    parser.add_argument("--mask_p", default=0.15, type=float, help="Masking probability.")
    parser.add_argument("--short_p", default=0.1, type=float, help="Short sequence probability.")
    parser.add_argument("--weight_decay_start", default=0.02, type=float, help="Short sequence probability.")
    parser.add_argument("--weight_decay_end", default=0.2, type=float, help="Short sequence probability.")
    parser.add_argument("--max_gradient", default=2.0, type=float, help="Max value for gradient clipping.")
    parser.add_argument('--mixed_precision', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--activation_checkpointing', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    return args


@torch.no_grad()
def log_parameter_histograms(model, step):
    for name, param in model.named_parameters():
        wandb.log(
            {
                f"parameters/norm_{name}": torch.linalg.norm(param.data).cpu().item(),
                f"parameters/std_{name}": param.data.std().cpu().item(),
            },
            step=step,
            commit=False
        )
        if param.requires_grad and param.grad is not None:
            wandb.log(
                {
                    f"gradients/norm_{name}": torch.linalg.norm(param.grad).cpu().item(),
                    f"gradients/std_{name}": param.grad.std().cpu().item(),
                },
                step=step,
                commit=False
            )


def setup_training(args):
    assert torch.cuda.is_available()
    args.n_gpu = torch.cuda.device_count()

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    seed_everything(args.seed + rank)

    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    if rank == 0:
        print(f"Group initialized? {torch.distributed.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(f"RCCL started on device {device}", flush=True)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    if is_main_process():
        os.system(f"mkdir -p {args.output_dir}")

    if is_main_process():
        print(f"Training for {args.max_steps:,} steps with {get_world_size()} GPUs")
        print(f"In total, the model will be trained on 'steps'({args.max_steps:,}) x 'GPUs'({get_world_size()}) x 'batch_size'({args.batch_size:,}) x 'seq_len'({args.seq_length:,}) = {args.max_steps * get_world_size() * args.batch_size * args.seq_length:,} subword instances")

    args.device_max_steps = args.max_steps

    if is_main_process():
        wandb.init(
            name="MAE_BYOL_MLM_only",
            config=args,
            id=args.wandb_id,
            project="BABY-LM",
            entity="ltg",
            resume="auto",
            tags=["BYOL", "base"],
            allow_val_change=True,
            reinit=True
        )

    return device, local_rank


def prepare_model_and_optimizer(args, device, local_rank, checkpoint):
    config = BertConfig(args.config_file)
    model = Bert(config, args.activation_checkpointing)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(config.to_dict())
        wandb.config.update({"n_params": n_params})
        print(model)
        print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=False)

    model.to(device)

    no_decay = ['bias', 'layer_norm', 'embedding']
    decay_params = [(n, p) for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad]
    no_decay_params = [(n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad]
    optimizer_grouped_parameters = [
        {'params': [p for _, p in decay_params], 'weight_decay': args.weight_decay_start},
        {'params': [p for _, p in no_decay_params], 'weight_decay': 0.0},
    ]

    if is_main_process():
        print("Parameters without weight decay:")
        for n, _ in no_decay_params:
            print(n)
        print()
        print("Parameters with weight decay:")
        for n, _ in decay_params:
            print(n)
        print(flush=True)

    if args.optimizer == "adam" or args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )
    elif args.optimizer == "lamb":
        optimizer = Lamb(
            optimizer_grouped_parameters,
            args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )
 
    scheduler = cosine_schedule_with_warmup(optimizer, int(args.device_max_steps * args.warmup_proportion), args.device_max_steps, 0.1)
    wd_scheduler = CosineWDSchedule(optimizer, args.weight_decay_start, args.device_max_steps, args.weight_decay_end)

    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        bucket_cap_mb=torch.cuda.get_device_properties(device).total_memory,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=True
    )

    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision, growth_interval=8000)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        grad_scaler.load_state_dict(checkpoint["grad_scaler"])
        wd_scheduler._step = checkpoint["global_step"]

    return model, config, optimizer, scheduler, wd_scheduler, grad_scaler


def training_epoch(model, tokenizer, data, optimizer, scheduler, wd_scheduler, grad_scaler, global_step, epoch, args, device, max_local_steps):
    train_dataloader = create_train_dataloader(data, tokenizer, args, global_step, args.seed + get_rank() + epoch * get_world_size())

    model = model.train()
    optimizer.zero_grad(set_to_none=True)

    if is_main_process():
        train_iter = tqdm(train_dataloader, desc="Train iteration", initial=global_step, total=args.device_max_steps)
    else:
        train_iter = train_dataloader

    for local_step, batch in enumerate(train_iter):
        encoder_inputs, full_inputs, outputs, unmasked_ids, encoder_attention_mask, full_attention_mask = [t.to(device, non_blocking=True) for t in batch]

        mlm_weight = args.mlm_weight_start + (args.mlm_weight_end - args.mlm_weight_start) * global_step / (args.device_max_steps // 8)
        mlm_weight = max(args.mlm_weight_end, mlm_weight)

        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            if global_step % 100 == 0:
                mlm_loss, byol_loss, _ = model(encoder_inputs, full_inputs, outputs, unmasked_ids, encoder_attention_mask, full_attention_mask)
                (mlm_loss + 0.0*byol_loss).backward()
                mlm_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                optimizer.zero_grad(set_to_none=True)

                mlm_loss, byol_loss, _ = model(encoder_inputs, full_inputs, outputs, unmasked_ids, encoder_attention_mask, full_attention_mask)
                (byol_loss + 0.0*mlm_loss).backward()
                byol_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                optimizer.zero_grad(set_to_none=True)

                if is_main_process():
                    wandb.log(
                        {
                            "stats/mlm_grad_norm": mlm_grad_norm,
                            "stats/byol_grad_norm": byol_grad_norm,
                        },
                        step=global_step,
                        commit=False,
                    )

            mlm_loss, byol_loss, accuracy = model(encoder_inputs, full_inputs, outputs, unmasked_ids, encoder_attention_mask, full_attention_mask)
            loss = mlm_weight * mlm_loss + byol_loss

        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient)

        return_value = grad_scaler.step(optimizer)
        grad_scaler.update()

        global_step += 1

        if return_value is None:
            optimizer.zero_grad(set_to_none=True)
            continue

        scheduler.step()
        wd_scheduler.step()

        with torch.no_grad():
            ema_decay = args.ema_decay_start + global_step * (args.ema_decay_end - args.ema_decay_start) / args.device_max_steps
            ema_decay = min(ema_decay, args.ema_decay_end)
            for param_q, param_k in zip(model.module.transformer.parameters(), model.module.teacher.parameters()):
                param_k.data.mul_(ema_decay).add_((1.0 - ema_decay) * param_q.detach().data)
            for param_q, param_k in zip(model.module.embedding.parameters(), model.module.teacher_embedding.parameters()):
                param_k.data.mul_(ema_decay).add_((1.0 - ema_decay) * param_q.detach().data)

            metrics = torch.stack([accuracy, mlm_loss, byol_loss, loss])
            torch.distributed.all_reduce(metrics, torch.distributed.ReduceOp.AVG)
            accuracy, mlm_loss, byol_loss, loss = metrics.tolist()

        if is_main_process():
            train_iter.set_postfix_str(f"loss: {loss:.2f}, accuracy: {accuracy * 100.0:.2f}, grad_norm: {grad_norm:.2f}, lr: {optimizer.param_groups[0]['lr']:.5f}")

            if global_step % 100 == 0:
                log_parameter_histograms(model, global_step)

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": loss,
                    "train/mlm_loss": mlm_loss,
                    "train/byol_loss": byol_loss,
                    "train/accuracy": accuracy * 100.0,
                    "stats/mlm_weight": mlm_weight,
                    "stats/learning_rate": optimizer.param_groups[0]['lr'],
                    "stats/weight_decay": optimizer.param_groups[0]['weight_decay'],
                    "stats/grad_norm": grad_norm,
                    "stats/seq_length": data.seq_length,
                    "stats/EMA_decay": ema_decay,
                    "stats/grad_scale": grad_scaler.get_scale(),
                },
                step=global_step,
            )

        optimizer.zero_grad(set_to_none=True)

        if global_step == int(args.device_max_steps * args.long_after):
            return global_step

        # Exiting the training due to hitting max steps
        if global_step >= args.device_max_steps or local_step >= max_local_steps - 1:
            return global_step

    return global_step


def save(model, optimizer, grad_scaler, scheduler, global_step, epoch, args):
    checkpoint_path = f"{args.output_dir}/model.bin"
    if is_main_process():
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
        torch.save(
            {
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "grad_scaler": grad_scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "args": args,
            },
            checkpoint_path
        )

    return checkpoint_path


def load_dataset(args, tokenizer, device):
    seq_length = args.seq_length * 4 if global_step >= int(args.device_max_steps * args.long_after) else args.seq_length
    train_data = Dataset(
        args.input_path.format(sequence_length=seq_length), get_rank(), get_world_size(), tokenizer, seq_length, args.mask_p, args.short_p
    )
    print(f"Loaded training file {get_rank()}", flush=True)

    batch_size = args.batch_size // 4 if global_step >= int(args.device_max_steps * args.long_after) else args.batch_size
    min_length = torch.tensor(len(train_data) // batch_size, dtype=torch.long, device=device)
    torch.distributed.all_reduce(min_length, torch.distributed.ReduceOp.MIN)

    if is_main_process():
        train_data.show_random_item()

    return train_data, min_length


def create_train_dataloader(data, tokenizer, args, global_step, seed):
    batch_size = args.batch_size // 4 if global_step >= int(args.device_max_steps * args.long_after) else args.batch_size
    train_dataloader = DataLoader(
        data,
        shuffle=True,
        batch_size=batch_size,
        num_workers=7 - 1,
        generator=torch.Generator().manual_seed(seed),
        drop_last=True,
        pin_memory=True,
        collate_fn=Collate(tokenizer.token_to_id("[PAD]"))
    )
    return train_dataloader


if __name__ == "__main__":
    args = parse_arguments()

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        checkpoint_args, initial_epoch, global_step = checkpoint["args"], checkpoint["epoch"] + 1, checkpoint["global_step"]
        args = vars(args).copy()
        args.update(vars(checkpoint_args))
        args = argparse.Namespace(**args)
    else:
        checkpoint, initial_epoch, global_step = None, 0, 0
        args.wandb_id = wandb.util.generate_id() if int(os.environ["SLURM_PROCID"]) == 0 else 0

    tokenizer = Tokenizer.from_file(args.vocab_path)
    device, local_rank = setup_training(args)
    model, config, optimizer, scheduler, wd_scheduler, grad_scaler = prepare_model_and_optimizer(args, device, local_rank, checkpoint)
    train_data, min_length = load_dataset(args, tokenizer, device)

    for epoch in count(initial_epoch):
        if global_step == int(args.device_max_steps * args.long_after):
            train_data, min_length = load_dataset(args, tokenizer, device)

        global_step = training_epoch(model, tokenizer, train_data, optimizer, scheduler, wd_scheduler, grad_scaler, global_step, epoch, args, device, min_length)
        checkpoint_path = save(model, optimizer, grad_scaler, scheduler, global_step, epoch, args)

        if global_step >= args.device_max_steps:
            break