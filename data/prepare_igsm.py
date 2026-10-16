"""Build the iGSM corpus (paper §4.2) in nanoGPT .bin format, plus the exact-operation eval sets.

Sequence layout:  problem <sol> solution <ans> answer <eos>

  * The vocabulary is compact: only ~2.1k of GPT-2's ids occur in iGSM text, so they are
    relabeled densely (a bijection on the same token stream, not a retokenization).
  * IdGen(op=15) realizes 8-15 operations, so exact-operation eval sets are made by generating
    and filtering on the realized count.
  * <sol>/<ans>/<eos> make the scoring boundary exact.

Requires the iGSM generator (MIT, Meta) and transformers (its tokenizer):
  git clone https://github.com/facebookresearch/iGSM.git third_party/iGSM
  pip install transformers
  python data/prepare_igsm.py --workers 16

The corpus depends on --workers as well as --seed (each worker gets its own seed). The paper's
corpus has these md5 sums; to train on exactly the same data, verify them or use the released copy:
  train.bin  8133cd5f9f72e15477cc31c396a8ba11
  val.bin    983d2c07aebfc9c9e744200bbfb555c6
  eval_op15.jsonl  a44d9eaada216b72848e366174786a4b
  meta.pkl   d22c2bbd6276b614c49e69d8c19fa0ac
"""
import argparse, json, os, pickle, random, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IGSM = os.path.join(_ROOT, "third_party", "iGSM")
_DEFAULT_OUT = os.path.join(_ROOT, "data", "igsm")
if _IGSM not in sys.path:
    sys.path.insert(0, _IGSM)

# Difficulty presets: "med" is the training distribution of Virtual Logic Depth (Zhu et al.);
# "wide" reaches deeper problems for out-of-distribution eval buckets.
PRESETS = {"med": dict(max_op=15, max_edge=20), "wide": dict(max_op=30, max_edge=40),
           "control": dict(max_op=30, max_edge=20)}
SPECIALS = ["<sol>", "<ans>", "<eos>"]


def _gen_chunk(job):
    """Worker: generate `n` problems, return [(realized_ops, prob, sol, ans)] as uint16."""
    seed, n, preset = job
    from data_gen.pretrain.id_gen import IdGen
    from tools.tools import tokenizer, fix_seed
    fix_seed(seed)
    random.seed(seed)
    cfg = PRESETS[preset]
    out = []
    for _ in range(n):
        g = IdGen(max_op=cfg["max_op"], max_edge=cfg["max_edge"], op=cfg["max_op"],
                  perm_level=5, detail_level=0)
        try:
            g.gen_prob([i for i in range(23)], p_format="pq")
        except Exception:
            continue  # rare generator dead-end; just draw another problem
        sol_txt = tokenizer.decode(g.sol_token)
        ops = len([s for s in sol_txt.split(".") if s.strip()])
        out.append((ops,
                    np.array(g.prob_token, dtype=np.uint16),
                    np.array(g.sol_token, dtype=np.uint16),
                    np.array(g.ans_token, dtype=np.uint16)))
    return out


def generate(n, preset, workers, seed0, want=None, cap_batches=200):
    """Generate until `n` problems satisfying `want(ops)` are collected (or we give up)."""
    from multiprocessing import Pool
    got, batches, seed = [], 0, seed0
    per = max(200, n // max(workers, 1))
    with Pool(workers) as pool:
        while len(got) < n and batches < cap_batches:
            jobs = [(seed + i, per, preset) for i in range(workers)]
            seed += workers
            for chunk in pool.imap_unordered(_gen_chunk, jobs):
                for rec in chunk:
                    if want is None or want(rec[0]):
                        got.append(rec)
            batches += 1
    return got[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=500_000)
    ap.add_argument("--n_eval", type=int, default=1000, help="per exact-op bucket")
    ap.add_argument("--val_frac", type=float, default=0.01)
    ap.add_argument("--test_frac", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_dir", default=_DEFAULT_OUT)
    args = ap.parse_args()
    HERE = args.out_dir
    os.makedirs(HERE, exist_ok=True)

    print(f"[1/4] train pool: {args.n_train} problems (ops<=15, VLD 'med' preset)")
    train = generate(args.n_train, "med", args.workers, args.seed, want=lambda o: o <= 15)
    print(f"      got {len(train)}")

    evals = {}
    for k, (bucket, preset) in enumerate({15: "med", 20: "wide", 21: "wide"}.items()):
        print(f"[2/4] eval bucket ops=={bucket} ({preset} preset), target {args.n_eval}")
        evals[bucket] = generate(args.n_eval, preset, args.workers, args.seed + 10_000 * (k + 1),
                                 want=lambda o, b=bucket: o == b)
        print(f"      got {len(evals[bucket])}")

    # Eval problems must not appear in train. Hash the problem tokens, not the text.
    train_keys = {r[1].tobytes() for r in train}
    for b, rows in evals.items():
        keep = [r for r in rows if r[1].tobytes() not in train_keys]
        if len(keep) != len(rows):
            print(f"      ops=={b}: dropped {len(rows)-len(keep)} problems that collide with train")
        evals[b] = keep

    print("[3/4] building compact vocab")
    used = set()
    for r in train:
        used.update(r[1].tolist()); used.update(r[2].tolist()); used.update(r[3].tolist())
    for rows in evals.values():
        for r in rows:
            used.update(r[1].tolist()); used.update(r[2].tolist()); used.update(r[3].tolist())
    gpt2_ids = sorted(used)
    remap = {g: i + len(SPECIALS) for i, g in enumerate(gpt2_ids)}
    SOL, ANS, EOS = range(len(SPECIALS))
    vocab_size = len(gpt2_ids) + len(SPECIALS)
    print(f"      {len(gpt2_ids)} GPT-2 ids used -> vocab_size {vocab_size}")

    def seq(rec):
        _, p, s, a = rec
        return ([remap[x] for x in p.tolist()] + [SOL] + [remap[x] for x in s.tolist()]
                + [ANS] + [remap[x] for x in a.tolist()] + [EOS])

    print("[4/4] writing")
    random.Random(args.seed).shuffle(train)
    n_val = int(len(train) * args.val_frac)
    n_test = int(len(train) * args.test_frac)
    splits = {"val": train[:n_val], "test": train[n_val:n_val + n_test], "train": train[n_val + n_test:]}
    for name, rows in splits.items():
        # Stream in chunks. Materialising every token as one Python list costs ~28 bytes per
        # int, which at 285M training tokens is ~8GB of RAM for a 570MB file.
        n_tok, buf = 0, []
        with open(os.path.join(HERE, f"{name}.bin"), "wb") as fh:
            for r in rows:
                buf.extend(seq(r))
                if len(buf) >= 8_000_000:
                    np.array(buf, dtype=np.uint16).tofile(fh); n_tok += len(buf); buf = []
            if buf:
                np.array(buf, dtype=np.uint16).tofile(fh); n_tok += len(buf)
        print(f"      {name}.bin: {len(rows)} problems, {n_tok:,} tokens")

    for b, rows in evals.items():
        path = os.path.join(HERE, f"eval_op{b}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                _, p, s, a = r
                f.write(json.dumps({
                    "prompt": [remap[x] for x in p.tolist()] + [SOL],
                    "solution": [remap[x] for x in s.tolist()] + [ANS],
                    "answer": [remap[x] for x in a.tolist()] + [EOS],
                }) + "\n")
        print(f"      eval_op{b}.jsonl: {len(rows)} problems")

    with open(os.path.join(HERE, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "gpt2_ids": gpt2_ids,
                     "specials": {"sol": SOL, "ans": ANS, "eos": EOS}}, f)
    print(f"done. vocab_size={vocab_size}")


if __name__ == "__main__":
    main()
