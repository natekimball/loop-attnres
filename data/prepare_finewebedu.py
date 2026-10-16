"""Prepare the FineWeb-Edu data used for language modeling (paper §4.1).

Streams FineWeb-Edu (no full download), tokenizes with the GPT-2 BPE (tiktoken),
and writes train.bin / val.bin / test.bin as uint16 in nanoGPT's format. Writes
incrementally (bounded memory). The first test_frac of documents go to test, the next
val_frac to val, and the rest to train (test/val are held out and disjoint from train).

  python data/prepare_finewebedu.py     # 500,000 documents, ~537M tokens, as in the paper
"""
import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finewebedu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_docs", type=int, default=500_000)
    ap.add_argument("--val_frac", type=float, default=0.005)
    ap.add_argument("--test_frac", type=float, default=0.005)
    ap.add_argument("--config", type=str, default="sample-10BT")
    ap.add_argument("--flush_tokens", type=int, default=5_000_000)
    args = ap.parse_args()

    os.makedirs(HERE, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=args.config, split="train", streaming=True)

    test_docs = max(1, int(args.num_docs * args.test_frac))
    val_docs = max(1, int(args.num_docs * args.val_frac))
    ftest = open(os.path.join(HERE, "test.bin"), "wb")
    fval = open(os.path.join(HERE, "val.bin"), "wb")
    ftrain = open(os.path.join(HERE, "train.bin"), "wb")
    buf_test, buf_val, buf_train = [], [], []
    n_test = n_val = n_train = 0

    def flush(buf, f):
        if buf:
            np.array(buf, dtype=np.uint16).tofile(f)
            buf.clear()

    n = 0
    for doc in ds:
        toks = enc.encode_ordinary(doc["text"])
        toks.append(eot)
        if n < test_docs:
            buf_test.extend(toks); n_test += len(toks)
            if len(buf_test) >= args.flush_tokens:
                flush(buf_test, ftest)
        elif n < test_docs + val_docs:
            buf_val.extend(toks); n_val += len(toks)
            if len(buf_val) >= args.flush_tokens:
                flush(buf_val, fval)
        else:
            buf_train.extend(toks); n_train += len(toks)
            if len(buf_train) >= args.flush_tokens:
                flush(buf_train, ftrain)
        n += 1
        if n % 20000 == 0:
            print(f"  {n}/{args.num_docs} docs | train {n_train:,} tok, val {n_val:,} tok, test {n_test:,} tok", flush=True)
        if n >= args.num_docs:
            break

    flush(buf_test, ftest); flush(buf_val, fval); flush(buf_train, ftrain)
    ftest.close(); fval.close(); ftrain.close()
    print(f"done: {n_train:,} train tokens, {n_val:,} val tokens, {n_test:,} test tokens (vocab=gpt2/50257)", flush=True)
    # the data is on disk; skip interpreter finalization, where streaming threads can fault
    os._exit(0)


if __name__ == "__main__":
    main()
