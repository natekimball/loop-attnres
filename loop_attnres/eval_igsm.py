"""Score a checkpoint on iGSM by greedy generation with exact answer match (paper §4.2, §5.7).

For each problem, the prompt ends with <sol>; the model decodes greedily until it emits <ans>,
and the next token is its answer, compared with the gold answer token. Every model is scored on
the same problems in the same order, so per-problem outcomes (--per_item_out) are paired across
models (scripts/paired_analysis.py).

    python -m loop_attnres.eval_igsm --ckpt out/ckpt.pt --data data/igsm --buckets 15 \
        --max_examples 1000 --kv_cache --json_out results/x.json --per_item_out results/x.items.jsonl

Looped models have no KV cache, so each decoded token reruns every iteration over the whole
sequence (about 50 s per problem at T = 16 on an A40). Dense models decode with a KV cache
(--kv_cache), which makes identical decisions.
"""
import argparse, json, os, time

import torch
import torch.nn.functional as F

from loop_attnres.nanogpt.model import GPT, GPTConfig
from loop_attnres.model import LoopedGPT, LoopedGPTConfig

SOL, ANS, EOS = 0, 1, 2


def load_model(ckpt_path, device):
    """Model from a training checkpoint; returns (model, is_looped, iteration)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["model_args"]
    looped = "n_loops" in a
    cls, cfg = (LoopedGPT, LoopedGPTConfig) if looped else (GPT, GPTConfig)
    model = cls(cfg(**{k: v for k, v in a.items() if k in cfg.__dataclass_fields__}))
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict({k: v for k, v in sd.items() if k in model.state_dict()})
    return model.eval().to(device), looped, ckpt.get("iter_num", -1)


@torch.no_grad()
def greedy_answer(model, prompt_ids, device, block_size, max_new):
    """Decode greedily to <ans>; return (predicted answer token or None, decode steps)."""
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]
    for step in range(max_new):
        logits, _ = model(idx if idx.size(1) <= block_size else idx[:, -block_size:])
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        tok = int(nxt.item())
        idx = torch.cat([idx, nxt], dim=1)
        if tok == ANS:
            logits, _ = model(idx if idx.size(1) <= block_size else idx[:, -block_size:])
            return int(logits[:, -1, :].argmax(-1).item()), step + 1
        if tok == EOS:
            return None, step + 1
    return None, max_new


@torch.no_grad()
def _gpt_forward_cached(model, idx, pos0, cache):
    """Dense GPT forward over new tokens with a per-layer key/value cache."""
    tr = model.transformer
    pos = torch.arange(pos0, pos0 + idx.size(1), device=idx.device)
    x = tr.wte(idx) + tr.wpe(pos)
    new = []
    for li, blk in enumerate(tr.h):
        a = blk.attn
        h = blk.ln_1(x)
        B, T, C = h.shape
        nh = a.n_head
        q, k, v = a.c_attn(h).split(a.n_embd, dim=2)
        q, k, v = (z.view(B, T, nh, C // nh).transpose(1, 2) for z in (q, k, v))
        if cache is not None:
            k = torch.cat([cache[li][0], k], 2); v = torch.cat([cache[li][1], v], 2)
        new.append((k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None)
        x = x + a.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))
        x = x + blk.mlp(blk.ln_2(x))
    return model.lm_head(tr.ln_f(x)[:, [-1], :]), new


@torch.no_grad()
def greedy_answer_cached(model, prompt_ids, device, block_size, max_new):
    """greedy_answer for a dense GPT with a KV cache (same decisions)."""
    if len(prompt_ids) + max_new + 1 > block_size:
        return greedy_answer(model, prompt_ids, device, block_size, max_new)
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]
    logits, cache = _gpt_forward_cached(model, idx, 0, None)
    n = idx.size(1)
    for step in range(max_new):
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        tok = int(nxt.item())
        logits, cache = _gpt_forward_cached(model, nxt, n, cache); n += 1
        if tok == ANS:
            return int(logits[:, -1, :].argmax(-1).item()), step + 1
        if tok == EOS:
            return None, step + 1
    return None, max_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/igsm")
    ap.add_argument("--buckets", default="15")
    ap.add_argument("--max_examples", type=int, default=1000)
    ap.add_argument("--max_new", type=int, default=700, help="decode cap")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--kv_cache", action="store_true", help="dense models: decode with a KV cache")
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--per_item_out", default=None,
                    help="per-problem outcomes (JSONL); appended as scored, and resumed from")
    args = ap.parse_args()

    model, looped, it = load_model(args.ckpt, args.device)
    bs = model.config.block_size
    print(f"{args.ckpt}: {'T=%d' % model.config.n_loops if looped else 'dense'}, iteration {it}")
    done = {}
    if args.per_item_out and os.path.exists(args.per_item_out):
        for l in open(args.per_item_out):
            if l.strip():
                r = json.loads(l); done[(r["bucket"], r["i"])] = r
    live = open(args.per_item_out, "a") if args.per_item_out else None

    results, per_item = {}, []
    for b in [x.strip() for x in args.buckets.split(",") if x.strip()]:
        rows = [json.loads(l) for l in open(os.path.join(args.data, f"eval_op{b}.jsonl"))][:args.max_examples]
        hit = emitted = 0
        steps_used, t0 = [], time.time()
        for i, r in enumerate(rows):
            gold = r["answer"][0]
            if (b, i) in done:
                pred, steps = done[(b, i)]["pred"], done[(b, i)]["steps"]
            else:
                decode = greedy_answer_cached if (args.kv_cache and not looped) else greedy_answer
                pred, steps = decode(model, r["prompt"], args.device, bs, args.max_new)
            ok = int(pred is not None and pred == gold)
            emitted += pred is not None
            hit += ok
            steps_used.append(steps)
            rec = {"bucket": b, "i": i, "gold": gold, "pred": pred, "correct": ok, "steps": steps}
            per_item.append(rec)
            if live is not None and (b, i) not in done:
                live.write(json.dumps(rec) + "\n"); live.flush()
            if (i + 1) % 25 == 0:
                print(f"  ops={b}: {i + 1}/{len(rows)} scored, {hit} correct, {time.time() - t0:.0f}s", flush=True)
        n = len(rows)
        at_cap = sum(s >= args.max_new for s in steps_used)
        results[b] = {"n": n, "answer_exact": hit / max(n, 1), "emitted_ans": emitted / max(n, 1),
                      "at_cap": at_cap}
        print(f"  ops={b}: accuracy {100 * hit / max(n, 1):.1f}%, reached <ans> {emitted}/{n}, "
              f"at decode cap {at_cap}/{n}")
    if live is not None:
        live.close()
        with open(args.per_item_out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in per_item)
    if args.json_out:
        json.dump(results, open(args.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
