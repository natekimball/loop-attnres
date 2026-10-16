"""Basic properties of the looped model (CPU; run with `pytest tests`)."""
import contextlib, io

import pytest
import torch

from loop_attnres.nanogpt.model import GPT, GPTConfig
from loop_attnres.model import LoopedGPT, LoopedGPTConfig

BASE = dict(n_layer=2, n_head=2, n_embd=64, block_size=16, vocab_size=97, dropout=0.0, bias=False)


def build(**kw):
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        m = LoopedGPT(LoopedGPTConfig(**BASE, **kw))
    torch.manual_seed(1)
    with torch.no_grad():                    # move off the zero-query initialization
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m.eval()


def batch():
    torch.manual_seed(2)
    return torch.randint(0, 97, (2, 16)), torch.randint(0, 97, (2, 16))


def test_one_iteration_identity_is_a_dense_gpt():
    m = build(n_loops=1, loop_residual="identity", deep_supervision=False)
    with contextlib.redirect_stdout(io.StringIO()):
        g = GPT(GPTConfig(**BASE))
    g.load_state_dict(m.state_dict())
    x, y = batch()
    assert torch.allclose(m(x, y)[1], g(x, y)[1], atol=1e-6)


@pytest.mark.parametrize("variant", [dict(attn_full_entrywise=True), dict(attn_carry_ckpt=True)])
def test_full_placement_memory_variants_compute_the_same_function(variant):
    ref = build(n_loops=3, loop_residual="attnres", attn_placement="full")
    alt = build(n_loops=3, loop_residual="attnres", attn_placement="full", **variant)
    alt.load_state_dict(ref.state_dict())
    x, y = batch()
    ref.train(); alt.train()
    assert torch.allclose(ref(x, y)[1], alt(x, y)[1], atol=1e-5)


def test_carry_is_a_convex_combination_of_the_history():
    m = build(n_loops=3, loop_residual="attnres")
    hist = torch.randn(2, 16, 4, 64)
    out, w = m.attn_res(hist, hist[:, :, -1], return_weights=True)
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)
    assert torch.allclose(out, torch.einsum("bshm,bsmd->bsd", w, hist), atol=1e-5)


def test_zero_query_starts_at_the_uniform_average():
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        m = LoopedGPT(LoopedGPTConfig(**BASE, n_loops=3, loop_residual="attnres"))
    hist = torch.randn(2, 16, 4, 64)
    assert torch.allclose(m.attn_res(hist, hist[:, :, -1]), hist.mean(2), atol=1e-6)


def test_scalar_carry_mixes_the_previous_state_and_the_embedding():
    m = build(n_loops=3, loop_residual="scalar")
    h0, h1 = torch.randn(2, 16, 64), torch.randn(2, 16, 64)
    assert torch.allclose(m._carry_in([h0, h1]), m.alpha * h1 + m.beta * h0)


def test_checkpoint_round_trip(tmp_path):
    from loop_attnres.eval_igsm import load_model
    m = build(n_loops=3, loop_residual="attnres", attn_placement="full")
    args = dict(BASE, n_loops=3, loop_residual="attnres", attn_placement="full")
    torch.save({"model": m.state_dict(), "model_args": args, "iter_num": 5}, tmp_path / "ckpt.pt")
    with contextlib.redirect_stdout(io.StringIO()):
        loaded, looped, it = load_model(str(tmp_path / "ckpt.pt"), "cpu")
    x, y = batch()
    assert looped and it == 5 and torch.equal(loaded(x, y)[1], m(x, y)[1])


def test_more_iterations_at_evaluation():
    m = build(n_loops=3, loop_residual="attnres")
    x, _ = batch()
    assert len(m.loop_logits(x, n_loops=6)) == 6
