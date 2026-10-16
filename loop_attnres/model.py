"""Looped transformer with Attention Residuals over the loop axis (LoopAttnRes).

A block of n_layer transformer layers is applied n_loops (T) times with shared weights. The
input to each application is formed by a *carry* from the states of earlier ones:

    h_0 = embed(x),   h_t = B(c_t),   c_t = C(h_0, ..., h_{t-1})

Carries (loop_residual):
    identity   c_t = h_{t-1}                       (plain layer repetition)
    scalar     c_t = alpha * h_{t-1} + beta * h_0  (two learned scalars)
    attnres    c   = sum_s softmax_s(q . RMSNorm(h_s) / sqrt(d)) h_s,   q = W_q h
               (Attention Residuals, Kimi Team 2026, applied across loop iterations)

Placement of the attention carry (attn_placement):
    loop   once per iteration, over the states at iteration boundaries
    full   before every layer, over the outputs of all earlier layers (all iterations)
    dual   once per iteration plus a second carry between the layers of each iteration
    sub    before every attention and MLP sublayer, over sublayer outputs (Kimi's granularity;
           attn_sub_group > 1 gives Block AttnRes)

Built on nanoGPT (vendored in loop_attnres/nanogpt); T = 1 is a standard dense transformer.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import functional as F

from loop_attnres.nanogpt.model import LayerNorm, Block, GPT, GPTConfig


@dataclass
class LoopedGPTConfig(GPTConfig):
    n_loops: int = 8                  # T, the number of iterations
    loop_residual: str = "identity"   # carry: "identity" | "scalar" | "attnres"
    loops_min: int = 0                # >0: sample T ~ U[loops_min, n_loops] per training step
    deep_supervision: bool = True     # LM loss at every iteration's output (else final only)
    deep_sup_subsample: int = 0       # >0: supervise the final iteration plus this many random others
    n_coda_layers: int = 0            # untied readout blocks between the loop and the head

    # attention carry
    attn_query: str = "input"         # "input" (q = W_q h) | "static" (one learned vector) |
                                      # "static_loop" (one learned vector per step) | "uniform"
    attn_key_norm: bool = True        # keys are RMS-normalized states
    attn_key_proj: bool = False
    attn_value_proj: bool = False
    attn_out_proj: bool = False       # implied by attn_heads > 1 or attn_value_proj
    attn_heads: int = 1
    attn_additive: bool = False       # c = h_{t-1} + attn(history) instead of c = attn(history)
    attn_carry_norm: bool = False     # LayerNorm on the history the carry reads (additive form)
    attn_carry_mlp: bool = False      # c = m + MLP(LN(m)), m = attn(history)
    attn_carry_mlp_ratio: int = 4
    attn_rel_pos_emb: bool = False    # learned bias on the scores by distance back on the loop axis
    attn_pos_emb: bool = False        # learned absolute embedding per history index, added to keys
    attn_placement: str = "loop"      # "loop" | "full" | "dual" | "sub"

    # query variants under full / sub placement
    attn_sub_query: str = "shared"    # "shared": one W_q for every carry
                                      # "per_iter": one W_q per iteration (loop, full and sub)
                                      # "per_layer": one W_q per sublayer position (sub)
                                      # "step_offset": shared W_q plus a learned vector per carry (sub)
                                      # "static_app": a learned vector per carry, no W_q (sub; Kimi's default)
    attn_sub_group: int = 1           # sub placement: sublayer outputs summed per history entry
                                      # (Block AttnRes block size; 1 = every sublayer output)
    attn_logit_scale: bool = True     # sub placement: divide carry logits by sqrt(d)
    attn_sub_qnorm: bool = False      # sub placement: query from RMSNorm(h)
    attn_q_lr_mult: float = 1.0       # learning-rate multiplier for the shared W_q

    # full placement memory options (same function, different memory/speed)
    attn_carry_ckpt: bool = False     # activation-checkpoint each carry
    attn_full_entrywise: bool = False # compute each carry entry by entry, keys normalized once
    attn_full_compiled: bool = False  # with attn_full_entrywise: compiled carry during training


def _rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _carry_core(q, scale, key_norm, *hs):
    """Full-placement carry over history entries hs, as one function for torch.compile: key
    norm, scores, softmax and weighted sum, with fp32 scores and fp32 accumulation."""
    qf = q.float()
    s = []
    for h in hs:
        k = h.float()
        if key_norm:
            k = _rms_norm(k)
        s.append((qf * k).sum(-1))
    w = torch.stack(s, dim=-1).mul(scale).softmax(-1)
    out = w[..., 0:1] * hs[0].float()
    for j in range(1, len(hs)):
        out = out + w[..., j:j + 1] * hs[j].float()
    return out.to(hs[-1].dtype)


# one graph per history length (static shapes); used in training only
_carry_core_compiled = torch.compile(_carry_core, dynamic=False)


class CarryMLP(nn.Module):
    """nanoGPT's MLP with a configurable expansion (c_proj gets the residual-scaled init)."""

    def __init__(self, config):
        super().__init__()
        d, h = config.n_embd, config.attn_carry_mlp_ratio * config.n_embd
        self.c_fc = nn.Linear(d, h, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(h, d, bias=config.bias)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class LoopAttnResidual(nn.Module):
    """Softmax attention over a history of states, separately for each token."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.n_embd
        self.heads = config.attn_heads
        assert d % self.heads == 0, "n_embd must be divisible by attn_heads"
        self.dh = d // self.heads
        assert config.attn_query in ("input", "static", "static_loop", "uniform"), config.attn_query
        if config.attn_query == "input":
            self.q_proj = nn.Linear(d, d, bias=False)
        elif config.attn_query == "static":
            self.q_vec = nn.Parameter(torch.zeros(d))
        elif config.attn_query == "static_loop":
            # one learned query per step, indexed by history length
            self.q_table = nn.Parameter(torch.zeros(config.n_loops * config.n_layer + 1, d))
        self.k_proj = nn.Linear(d, d, bias=False) if config.attn_key_proj else None
        self.v_proj = nn.Linear(d, d, bias=False) if config.attn_value_proj else None
        need_o = self.heads > 1 or config.attn_value_proj or config.attn_out_proj
        self.o_proj = nn.Linear(d, d, bias=False) if need_o else None
        if config.attn_carry_mlp:
            self.ln_mlp = LayerNorm(d, bias=config.bias)
            self.mlp = CarryMLP(config)
        else:
            self.ln_mlp = self.mlp = None
        self.ln_carry = LayerNorm(d, bias=config.bias) if config.attn_carry_norm else None
        # learned bias by distance back on the loop axis (zero-init); distances past the
        # trained history clamp to the last bucket
        self.rel_pos_bias = (nn.Parameter(torch.zeros(self.heads, config.n_loops * config.n_layer + 1))
                             if config.attn_rel_pos_emb else None)
        # learned absolute embedding per history index, added to the keys (zero-init)
        if config.attn_pos_emb:
            self.pos_emb = nn.Embedding(config.n_loops * config.n_layer + 1, d)
            torch.nn.init.zeros_(self.pos_emb.weight)
        else:
            self.pos_emb = None

    def _post(self, out):
        if self.mlp is not None:
            out = out + self.mlp(self.ln_mlp(out))
        return out

    def forward(self, history, latest, return_weights=False, q_proj=None):
        """history: [B, S, M, D] states; latest: [B, S, D] the most recent one (query input).
        q_proj overrides the module's W_q (per-iteration queries). Returns [B, S, D] (and the
        weights [B, S, H, M] if return_weights)."""
        cfg = self.config
        B, S, M, D = history.shape
        H, dh = self.heads, self.dh
        q_src = latest
        if self.ln_carry is not None:
            history = self.ln_carry(history)
            q_src = self.ln_carry(latest)

        v = self.v_proj(history) if self.v_proj is not None else history
        if cfg.attn_query == "uniform":
            out = v.mean(dim=2)
            out = self.o_proj(out) if self.o_proj is not None else out
            out = self._post(out)
            return (out, None) if return_weights else out

        k_src = history
        if self.pos_emb is not None:
            idx = torch.arange(M, device=history.device).clamp(max=self.pos_emb.num_embeddings - 1)
            k_src = history + self.pos_emb(idx).view(1, 1, M, D).to(history.dtype)
        k = self.k_proj(k_src) if self.k_proj is not None else k_src
        if cfg.attn_key_norm:
            k = _rms_norm(k)
        if cfg.attn_query == "input":
            q = (q_proj if q_proj is not None else self.q_proj)(q_src)
        elif cfg.attn_query == "static_loop":
            q = self.q_table[min(M - 1, self.q_table.shape[0] - 1)].expand(B, S, D)
        else:
            q = self.q_vec.expand(B, S, D)

        q = q.view(B, S, H, dh)
        k = k.view(B, S, M, H, dh)
        # scores in fp32 even under autocast
        with torch.autocast(device_type=q.device.type, enabled=False):
            scores = torch.einsum("bshd,bsmhd->bshm", q.float(), k.float()) * (dh ** -0.5)
        if self.rel_pos_bias is not None:
            d = torch.arange(M - 1, -1, -1, device=scores.device).clamp(max=self.rel_pos_bias.shape[1] - 1)
            scores = scores + self.rel_pos_bias[:, d].to(scores.dtype).view(1, 1, H, M)
        w = scores.softmax(dim=-1)
        vh = v.view(B, S, M, H, dh)
        out = torch.einsum("bshm,bsmhd->bshd", w.to(vh.dtype), vh).reshape(B, S, D)
        out = self.o_proj(out) if self.o_proj is not None else out
        out = self._post(out)
        return (out, w) if return_weights else out


class LoopedGPT(GPT):
    def __init__(self, config: LoopedGPTConfig):
        nn.Module.__init__(self)
        assert config.vocab_size is not None and config.block_size is not None
        assert config.loop_residual in ("identity", "scalar", "attnres"), config.loop_residual
        assert config.attn_placement in ("loop", "full", "dual", "sub"), config.attn_placement
        assert config.attn_placement == "loop" or config.loop_residual == "attnres", \
            "full, dual and sub placement apply to the attention carry"
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),   # the shared block
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        assert config.n_coda_layers >= 0
        self.coda = nn.ModuleList([Block(config) for _ in range(config.n_coda_layers)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        if config.loop_residual == "attnres":
            self.attn_res = LoopAttnResidual(config)
            if config.attn_placement == "dual":
                self.attn_res_intra = LoopAttnResidual(config)
        elif config.loop_residual == "scalar":
            self.alpha = nn.Parameter(torch.ones(()))
            self.beta = nn.Parameter(torch.ones(()))

        self.apply(self._init_weights)
        # residual branches are scaled by the effective depth L x T, not by L
        eff_depth = config.n_layer * config.n_loops
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * eff_depth))
        # the readout blocks run once, so they are scaled by their own depth
        for pn, p in self.coda.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_coda_layers))
        if config.loop_residual == "attnres":
            if config.attn_query == "input":
                # zero query: uniform attention over the history at initialization
                torch.nn.init.zeros_(self.attn_res.q_proj.weight)
                if config.attn_placement in ("full", "loop") and config.attn_sub_query == "per_iter":
                    self.sub_q = self._zero_queries(config.n_loops)
                if config.attn_placement == "sub":
                    sq = config.attn_sub_query
                    assert sq in ("shared", "per_layer", "per_iter", "step_offset", "static_app"), sq
                    n = {"per_layer": 2 * config.n_layer + 1, "per_iter": config.n_loops}.get(sq, 0)
                    if n:
                        self.sub_q = self._zero_queries(n)
                    if sq in ("step_offset", "static_app"):   # one per carry: 2L per iteration + readout
                        self.sub_b = nn.Parameter(torch.zeros((2 * config.n_layer + 1) * config.n_loops,
                                                              config.n_embd))
            for m in (self.attn_res, getattr(self, "attn_res_intra", None)):
                if m is None:
                    continue
                if m is not self.attn_res and config.attn_query == "input":
                    torch.nn.init.zeros_(m.q_proj.weight)
                # projections start at the identity (replacement) or zero (additive), so every
                # variant starts at exactly the plain carry
                if m.o_proj is not None:
                    (torch.nn.init.zeros_ if config.attn_additive else torch.nn.init.eye_)(m.o_proj.weight)
                if m.v_proj is not None:
                    torch.nn.init.eye_(m.v_proj.weight)
                if m.mlp is not None:
                    torch.nn.init.zeros_(m.mlp.c_proj.weight)
                    if m.mlp.c_proj.bias is not None:
                        torch.nn.init.zeros_(m.mlp.c_proj.bias)
        print("number of parameters: %.2fM (%d iterations x %d layers)" % (
            self.get_num_params() / 1e6, config.n_loops, config.n_layer))

    def _zero_queries(self, n):
        qs = nn.ModuleList([nn.Linear(self.config.n_embd, self.config.n_embd, bias=False) for _ in range(n)])
        for q in qs:
            torch.nn.init.zeros_(q.weight)
        return qs

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = super().configure_optimizers(weight_decay, learning_rate, betas, device_type)
        # an identity-initialized output projection decays toward the identity's opposite
        # (carry nothing), so it is excluded from weight decay in the replacement form
        if (self.config.loop_residual == "attnres" and not self.config.attn_additive
                and self.attn_res.o_proj is not None):
            w = self.attn_res.o_proj.weight
            decay_g = next((g for g in optimizer.param_groups if g["weight_decay"] > 0), None)
            nodecay_g = next((g for g in optimizer.param_groups if g["weight_decay"] == 0), None)
            if decay_g is not None and nodecay_g is not None and any(p is w for p in decay_g["params"]):
                decay_g["params"] = [p for p in decay_g["params"] if p is not w]
                nodecay_g["params"].append(w)
        if self.config.attn_q_lr_mult != 1.0 and self.config.loop_residual == "attnres":
            w = self.attn_res.q_proj.weight
            src = next(g for g in optimizer.param_groups if any(p is w for p in g["params"]))
            src["params"] = [p for p in src["params"] if p is not w]
            optimizer.add_param_group({"params": [w], "weight_decay": src["weight_decay"],
                                       "lr_mult": self.config.attn_q_lr_mult})
        return optimizer

    # ---- readout ------------------------------------------------------------------------------
    def _readout(self, z):
        """Decode-ready state from an iteration's output: readout blocks, then ln_f."""
        for block in self.coda:
            z = block(z)
        return self.transformer.ln_f(z)

    # ---- carries ------------------------------------------------------------------------------
    def _apply_res(self, module, history, q_proj=None):
        out = (module(torch.stack(history, dim=2), history[-1]) if q_proj is None else
               module(torch.stack(history, dim=2), history[-1], q_proj=q_proj))
        if self.config.attn_additive:
            return history[-1] + out
        return out

    def _apply_res_flat(self, module, q_proj, *history):
        return self._apply_res(module, list(history), q_proj=q_proj)

    def _carry_in(self, history, q_proj=None):
        mode = self.config.loop_residual
        if mode == "identity":
            return history[-1]
        if mode == "scalar":
            return self.alpha * history[-1] + self.beta * history[0]
        return self._apply_res(self.attn_res, history, q_proj=q_proj)

    def _check_plain_carry(self, what):
        mod, cfg = self.attn_res, self.config
        assert mod.heads == 1 and mod.k_proj is None and mod.v_proj is None and mod.o_proj is None \
            and mod.mlp is None and mod.ln_carry is None and mod.pos_emb is None \
            and mod.rel_pos_bias is None and cfg.attn_query == "input" and not cfg.attn_additive, \
            f"{what}: supports the plain attention carry only"

    def _full_key(self, h):
        return _rms_norm(h) if self.config.attn_key_norm else h

    def _full_carry(self, history, keys, q_proj=None, compiled=False):
        """The carry over a list of entries without stacking them: fp32 scores entry by entry,
        softmax, fp32 weighted sum. Memory and traffic per carry are linear in the history."""
        self._check_plain_carry("attn_full_entrywise")
        mod, cfg = self.attn_res, self.config
        q = (q_proj if q_proj is not None else mod.q_proj)(history[-1])
        scale = mod.dh ** -0.5
        if compiled:
            return _carry_core_compiled(q, scale, cfg.attn_key_norm, *history)
        with torch.autocast(device_type=q.device.type, enabled=False):
            qf = q.float()
            w = torch.stack([(qf * k.float()).sum(-1) for k in keys], dim=-1).mul(scale).softmax(-1)
            out = w[..., 0:1] * history[0]
            for j in range(1, len(history)):
                out = out + w[..., j:j + 1] * history[j]
        return out.to(history[-1].dtype)

    def _sub_key(self, v):
        return _rms_norm(v.float()) if self.config.attn_key_norm else v.float()

    def _sub_carry(self, vals, keys, latest, i=0, pos=0):
        """Carry under sublayer placement over a list of entries (i: iteration, pos: carry
        position within the iteration, 2L = the iteration's output)."""
        self._check_plain_carry("sub placement")
        mod, cfg = self.attn_res, self.config
        with torch.autocast(device_type=latest.device.type, enabled=False):
            sq = cfg.attn_sub_query
            qp = (self.sub_q[pos] if sq == "per_layer" else
                  self.sub_q[min(i, len(self.sub_q) - 1)] if sq == "per_iter" else mod.q_proj)
            if sq == "static_app":
                k = min(i * (2 * len(self.transformer.h) + 1) + pos, self.sub_b.shape[0] - 1)
                q = self.sub_b[k].float().expand(latest.shape)
            else:
                q = qp(_rms_norm(latest.float()) if cfg.attn_sub_qnorm else latest).float()
            if sq == "step_offset":
                k = min(i * (2 * len(self.transformer.h) + 1) + pos, self.sub_b.shape[0] - 1)
                q = q + self.sub_b[k].float()
            if cfg.attn_logit_scale:
                q = q * (latest.shape[-1] ** -0.5)
            w = torch.stack([(q * k).sum(-1) for k in keys], dim=-1).softmax(-1)
        out = w[..., 0:1].to(vals[0].dtype) * vals[0]
        for j in range(1, len(vals)):
            out = out + w[..., j:j + 1].to(vals[j].dtype) * vals[j]
        return out

    # ---- the loop -------------------------------------------------------------------------------
    def _sample_n_loops(self):
        if self.training and self.config.loops_min > 0:
            return int(torch.randint(self.config.loops_min, self.config.n_loops + 1, ()).item())
        return self.config.n_loops

    def _run_loops(self, idx, n_loops=None):
        """Raw (pre-readout) output state of every iteration. n_loops overrides config.n_loops
        (e.g. to evaluate at more iterations than trained)."""
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x0 = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        T = n_loops if n_loops is not None else self.config.n_loops
        cfg = self.config
        placement = cfg.attn_placement if cfg.loop_residual == "attnres" else "loop"
        history, outs = [x0], []
        sub_q = getattr(self, "sub_q", None)

        if placement == "sub":
            # values are sublayer outputs, v_0 = the embedding; every sublayer's input is the
            # carry's mixture of earlier values, with no residual addition
            vals, keys, h = [x0], [self._sub_key(x0)], x0
            G = max(1, cfg.attn_sub_group)
            part = [None, 0]                  # current group's partial sum and its size
            L = len(self.transformer.h)

            def carry(h, i, p):
                if part[0] is None:
                    return self._sub_carry(vals, keys, h, i, p)
                return self._sub_carry(vals + [part[0]], keys + [self._sub_key(part[0])], h, i, p)

            def emit(o):
                part[0] = o if part[0] is None else part[0] + o
                part[1] += 1
                if part[1] == G:              # a complete group becomes a history entry
                    vals.append(part[0]); keys.append(self._sub_key(part[0])); part[0], part[1] = None, 0

            for i in range(T):
                for li, block in enumerate(self.transformer.h):
                    h = carry(h, i, 2 * li)
                    emit(block.attn(block.ln_1(h)))
                    h = carry(h, i, 2 * li + 1)
                    emit(block.mlp(block.ln_2(h)))
                h = carry(h, i, 2 * L)
                outs.append(h)
            return outs

        if placement == "full":
            keys = []   # attn_full_entrywise: one normalized key per history entry
            for i in range(T):
                qp = sub_q[min(i, len(sub_q) - 1)] if sub_q is not None else None
                for block in self.transformer.h:
                    if cfg.attn_full_entrywise and cfg.attn_full_compiled and torch.is_grad_enabled():
                        z = self._full_carry(history, None, qp, compiled=True)
                    elif cfg.attn_full_entrywise:
                        if len(keys) < len(history):
                            keys.extend(self._full_key(h) for h in history[len(keys):])
                        z = self._full_carry(history, keys, qp)
                    elif cfg.attn_carry_ckpt and torch.is_grad_enabled():
                        z = torch.utils.checkpoint.checkpoint(
                            self._apply_res_flat, self.attn_res, qp, *history, use_reentrant=False)
                    else:
                        z = self._apply_res(self.attn_res, history, q_proj=qp)
                    z = block(z)
                    history.append(z)
                outs.append(z)
            return outs

        if placement == "dual":
            # a carry at each iteration boundary, plus one between the layers of each iteration
            # over a history that restarts every iteration
            for i in range(T):
                z = self._apply_res(self.attn_res, history)
                group_hist = [z]
                for block in self.transformer.h:
                    z = self._apply_res(self.attn_res_intra, group_hist)
                    z = block(z)
                    group_hist.append(z)
                history.append(z)
                outs.append(z)
            return outs

        for i in range(T):                    # loop placement
            qp = sub_q[min(i, len(sub_q) - 1)] if sub_q is not None else None
            z = self._carry_in(history, q_proj=qp)
            for block in self.transformer.h:
                z = block(z)
            history.append(z)
            outs.append(z)
        return outs

    def _supervised_loops(self, n_outs):
        if not self.config.deep_supervision:
            return [n_outs - 1]
        k = self.config.deep_sup_subsample
        if k <= 0 or not self.training or k >= n_outs - 1:
            return list(range(n_outs))
        return sorted(torch.randperm(n_outs - 1)[:k].tolist()) + [n_outs - 1]

    def forward(self, idx, targets=None):
        outs = self._run_loops(idx, n_loops=self._sample_n_loops())
        V = self.lm_head.out_features
        if targets is not None:
            logits = self.lm_head(self._readout(outs[-1]))
            supervised = self._supervised_loops(len(outs))
            losses = [F.cross_entropy(
                          (logits if i == len(outs) - 1 else self.lm_head(self._readout(outs[i]))).view(-1, V),
                          targets.view(-1), ignore_index=-1)
                      for i in supervised]
            return logits, torch.stack(losses).mean()
        return self.lm_head(self._readout(outs[-1])[:, [-1], :]), None

    @torch.no_grad()
    def loop_logits(self, idx, n_loops=None):
        """Logits (last position) after every iteration, as a list of [B, 1, V]."""
        return [self.lm_head(self._readout(o)[:, [-1], :]) for o in self._run_loops(idx, n_loops=n_loops)]
