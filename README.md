# LoopAttnRes

Code for **Attention Over Depth Makes Weight Tying Cheap** (Nathaniel Kimball, 2026).

A looped (weight-tied) transformer applies one block of `L` layers `T` times. LoopAttnRes
replaces the residual connection between iterations with Attention Residuals
([Kimi Team, 2026](https://arxiv.org/abs/2603.15031)): before each iteration, every token attends
over its own states from all earlier iterations, at the cost of one `d × d` query matrix.

```
c_t = Σ_{s<t} softmax_s( q_t · RMSNorm(h_s) / √d ) h_s,    q_t = W_q h_{t-1},    h_t = Block(c_t)
```

The carry can be applied once per iteration (*loop placement*) or before every layer, over the
outputs of all earlier layers (*full placement*).

## Install

```bash
pip install torch          # the build matching your CUDA version
pip install -r requirements.txt
```

## Data

```bash
python data/prepare_finewebedu.py          # FineWeb-Edu, 500,000 documents (~537M tokens)

git clone https://github.com/facebookresearch/iGSM.git third_party/iGSM
pip install transformers
python data/prepare_igsm.py --workers 16    # iGSM training corpus and eval sets
```

`data/igsm/eval_op15.jsonl` and `data/igsm/meta.pkl` are included: they are the 1,000
15-operation problems every iGSM model in the paper is scored on, and the vocabulary they are
encoded with. The generated corpus depends on `--workers` as well as the seed;
`data/prepare_igsm.py` lists the md5 sums of the paper's files.

## Train

```bash
# attention carry, loop placement, a 5-layer block looped 8 times (FineWeb-Edu)
python -m loop_attnres.train --dataset=finewebedu --n_layer=5 --n_loops=8

# full placement
python -m loop_attnres.train --dataset=finewebedu --n_layer=5 --n_loops=8 --attn_placement=full

# dense baseline (n_loops=0) and the identity carry (plain layer repetition)
python -m loop_attnres.train --dataset=finewebedu --n_layer=40 --n_loops=0
python -m loop_attnres.train --dataset=finewebedu --n_layer=5 --n_loops=8 --loop_residual=identity
```

Defaults: `d = 512`, 8 heads, context 512, effective batch 256 sequences
(`batch_size × gradient_accumulation_steps`), 20,000 iterations with cosine decay. On smaller
GPUs keep the effective batch at 256, e.g. `--batch_size=8 --gradient_accumulation_steps=32`.
Full placement at many iterations is memory-heavy; `--attn_full_entrywise=True
--attn_full_compiled=True --compile=True --compile_blocks=True` computes the same function with
memory linear in the history.

### The paper's models

[`configs/runs.csv`](configs/runs.csv) lists every model reported in the paper (88 rows): where
it appears, its result (lowest validation loss and perplexity, or iGSM accuracy), and the
command that trains it with this code. For example:

```bash
python -m loop_attnres.train --dataset=igsm --n_layer=5 --n_loops=16 --attn_placement=full \
    --attn_full_entrywise=True --attn_full_compiled=True --batch_size=8 --block_size=2048 \
    --gradient_accumulation_steps=32 --lr_decay_iters=6000 --max_iters=6000
```

## Evaluate on iGSM

Greedy decoding with exact answer match, on the same 1,000 problems as the paper:

```bash
python -m loop_attnres.eval_igsm --ckpt out/ckpt.pt --data data/igsm --buckets 15 \
    --max_examples 1000 --json_out results/x.json --per_item_out results/x.items.jsonl
```

Looped models have no KV cache, so scoring is slow (about 50 s per problem at `T = 16` on an
A40). Dense models can add `--kv_cache`.

## Reproduce the paper's analyses

```bash
python scripts/paired_analysis.py   # Tables 9-11: paired iGSM comparisons and growth tests
python scripts/param_ratios.py      # §5.1, Table 8 and Appendix D: dense parameters needed
python scripts/make_figures.py      # Figures 1 and 2
```

[`results/igsm/`](results/igsm) holds the per-problem outcomes of all 23 iGSM models, which these
scripts read. `param_ratios.py` counts parameters exactly, so a few ratios differ from the paper
in the last digit.

## Layout

```
loop_attnres/model.py       the looped model and its carries and placements
loop_attnres/train.py       training (nanoGPT style)
loop_attnres/eval_igsm.py   iGSM scoring
loop_attnres/nanogpt/       nanoGPT model and configurator (unmodified, MIT)
data/                       data preparation; iGSM eval set
configs/runs.csv            the paper's models
results/igsm/               per-problem iGSM outcomes
scripts/                    analyses and figures
tests/                      pytest tests
```

## Citation

```bibtex
@article{kimball2026loopattnres,
  title  = {Attention Over Depth Makes Weight Tying Cheap},
  author = {Kimball, Nathaniel},
  year   = {2026}
}
```

## License

MIT. `loop_attnres/nanogpt/` is from [nanoGPT](https://github.com/karpathy/nanoGPT) (MIT,
Andrej Karpathy). The iGSM generator ([facebookresearch/iGSM](https://github.com/facebookresearch/iGSM),
MIT) is not included.
