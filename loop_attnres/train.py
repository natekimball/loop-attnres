"""Train a dense or looped GPT (single GPU), nanoGPT style.

    python -m loop_attnres.train --dataset=finewebedu --n_layer=5 --n_loops=8 --loop_residual=attnres

n_loops=0 trains a standard (dense) GPT. Any setting below can be overridden with --key=value or
a config .py file (nanoGPT's configurator); configs/runs.csv lists the settings of every model
reported in the paper. Checkpoints are selected on the validation loss of the final iteration.
"""
import os
import sys
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from loop_attnres.nanogpt.model import GPTConfig, GPT
from loop_attnres.model import LoopedGPTConfig, LoopedGPT

# ----- defaults -----
out_dir = 'out'
eval_interval = 500
log_interval = 10
eval_iters = 100
eval_only = False
abort_val_rise = 0.0             # >0: stop (exit code 3) if val loss rises this far above the best
init_from = 'scratch'            # 'scratch' | 'resume'
wandb_log = False
wandb_project = 'loop-attnres'
wandb_run_name = 'run'
dataset = 'finewebedu'           # data/<dataset>/{train,val,test}.bin
gradient_accumulation_steps = 8
batch_size = 32                  # effective batch = batch_size x gradient_accumulation_steps (256)
block_size = 512
# model (n_layer = layers in the shared block when looped)
n_layer = 5
n_head = 8
n_embd = 512
dropout = 0.0
bias = False
# looping
n_loops = 0                      # 0 => dense GPT
loop_residual = 'attnres'        # identity | scalar | attnres
loops_min = 0
deep_supervision = False
deep_sup_subsample = 0
n_coda_layers = 0
attn_query = 'input'
attn_key_norm = True
attn_key_proj = False
attn_value_proj = False
attn_out_proj = False
attn_heads = 1
attn_additive = False
attn_carry_norm = False
attn_carry_mlp = False
attn_carry_mlp_ratio = 4
attn_rel_pos_emb = False
attn_pos_emb = False
attn_placement = 'loop'          # loop | full | dual | sub
attn_sub_query = 'shared'
attn_sub_group = 1
attn_logit_scale = True
attn_sub_qnorm = False
attn_q_lr_mult = 1.0
attn_carry_ckpt = False
attn_full_entrywise = False
attn_full_compiled = False
final_test = True                # evaluate the test split once at the end, on the best checkpoint
eval_seed = 2357                 # fixed evaluation batches, identical across runs
# optimizer
learning_rate = 6e-4
max_iters = 20000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 200
lr_decay_iters = 20000
min_lr = 6e-5
# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
compile_blocks = False           # with compile: compile each block and run the carry eagerly
seed = 1337
# ---------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(__file__), 'nanogpt', 'configurator.py')).read())
config = {k: globals()[k] for k in config_keys}
# ---------------------

is_looped = n_loops > 0
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', dataset)


def get_batch(split, generator=None):
    """A random batch; evaluation passes a seeded generator so it never touches the training RNG."""
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


iter_num = 0
best_val_loss = 1e9
meta_path = os.path.join(data_dir, 'meta.pkl')
vocab_size = 50304
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        vocab_size = pickle.load(f)['vocab_size']

if init_from == 'resume':
    ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
    model_args = ckpt['model_args']
    is_looped = 'n_loops' in model_args
    n_loops = model_args.get('n_loops', 0)
    config.update({k: v for k, v in model_args.items() if k in config})
else:
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, vocab_size=vocab_size, dropout=dropout)
    if is_looped:
        model_args.update({k: config[k] for k in LoopedGPTConfig.__dataclass_fields__
                           if k in config and k not in model_args})
model = LoopedGPT(LoopedGPTConfig(**model_args)) if is_looped else GPT(GPTConfig(**model_args))
if init_from == 'resume':
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()})
    iter_num, best_val_loss = ckpt['iter_num'], ckpt['best_val_loss']
model.to(device)

scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(ckpt['optimizer'])
    ckpt = None
raw_model = model
if compile:
    if is_looped and (loops_min > 0 or attn_placement != 'loop'):
        # one graph per sampled loop count / history length
        torch._dynamo.config.cache_size_limit = 128
    if compile_blocks:
        for blk in raw_model.transformer.h:
            blk.compile()
    else:
        model = torch.compile(model)


def eval_generator(split):
    g = torch.Generator()
    g.manual_seed(eval_seed + {'train': 0, 'val': 1, 'test': 2}[split])
    return g


@torch.no_grad()
def estimate_loss():
    """Train and val loss, and for looped models the val loss after every iteration."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        g = eval_generator(split)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, g)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    if is_looped:
        g = eval_generator('val')
        per_loop = torch.zeros(eval_iters, n_loops)
        for k in range(eval_iters):
            X, Y = get_batch('val', g)
            with ctx:
                for li, o in enumerate(raw_model._run_loops(X)):
                    lg = raw_model.lm_head(raw_model._readout(o))
                    per_loop[k, li] = torch.nn.functional.cross_entropy(
                        lg.view(-1, lg.size(-1)), Y.view(-1), ignore_index=-1).item()
        out['val_per_loop'] = per_loop.mean(0).tolist()
        out['val_lastloop'] = out['val_per_loop'][-1]
    model.train()
    return out


@torch.no_grad()
def estimate_split(split):
    model.eval()
    g = eval_generator(split)
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(split, g)
        with ctx:
            _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    r = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * r)) * (learning_rate - min_lr)


if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb.define_metric("iter")
    wandb.define_metric("*", step_metric="iter")

print(f"params={sum(p.numel() for p in raw_model.parameters()) / 1e6:.2f}M")


def save(path):
    torch.save({'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                'model_args': model_args, 'iter_num': iter_num,
                'best_val_loss': best_val_loss, 'config': config}, path + '.tmp')
    os.replace(path + '.tmp', path)


X, Y = get_batch('train')
t0 = time.time()
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1.0)

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        if abort_val_rise > 0 and (not math.isfinite(losses['val'])
                                   or losses['val'] > best_val_loss + abort_val_rise):
            print(f"ABORT: val {losses['val']:.4f} at step {iter_num} vs best {best_val_loss:.4f}", flush=True)
            sys.exit(3)
        msg = f"step {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}"
        if is_looped:
            msg += f", val_lastloop {losses['val_lastloop']:.4f} | per iteration [" + \
                   ", ".join(f"{v:.3f}" for v in losses['val_per_loop']) + "]"
        print(msg)
        if wandb_log:
            log = {"iter": iter_num, "train/loss": losses['train'], "val/loss": losses['val'], "lr": lr}
            if is_looped:
                log["val/loss_lastloop"] = losses['val_lastloop']
                log.update({f"val/loop_{li + 1}": v for li, v in enumerate(losses['val_per_loop'])})
            wandb.log(log)
        sel_loss = losses['val_lastloop'] if is_looped else losses['val']
        if sel_loss < best_val_loss:
            best_val_loss = sel_loss
            if iter_num > 0:
                save(os.path.join(out_dir, 'ckpt.pt'))
        if iter_num > 0:
            save(os.path.join(out_dir, 'ckpt_last.pt'))
    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if iter_num % log_interval == 0:
        dt = time.time() - t0
        t0 = time.time()
        print(f"iter {iter_num}: loss {loss.item() * gradient_accumulation_steps:.4f}, "
              f"{dt * 1000 / max(log_interval, 1):.1f}ms/iter")
    iter_num += 1
    if iter_num > max_iters:
        break

if final_test and os.path.exists(os.path.join(data_dir, 'test.bin')):
    best = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
    raw_model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in best['model'].items()})
    test_loss = estimate_split('test')
    print(f"test loss {test_loss:.4f} (ppl {math.exp(test_loss):.2f}), best checkpoint iter {best['iter_num']}")
    if wandb_log:
        wandb.log({"test/loss": test_loss, "test/ppl": math.exp(test_loss)})
