"""Autoregressive generation for trained causal-LM backbones.

The generative payoff of the library: sample new tokens from a model one step at a
time, or complete a design from a prefix (give a model a promoter+RBS+CDS and let
it propose the rest). Operates on a single sequence at a time, which keeps the
sampling code small and is all the CLI and design-completion need.

Plain causal generation extends a prefix; bidirectional infilling between a fixed
prefix and suffix needs fill-in-the-middle training and is a future extension.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from sboltorch.tokenize.base import Tokenizer


def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Apply top-k and nucleus (top-p) filtering, returning masked logits."""
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if 0.0 < top_p < 1.0:
        ordered, order = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(ordered, dim=-1), dim=-1)
        remove = cumulative - F.softmax(ordered, dim=-1) >= top_p
        ordered = ordered.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter(-1, order, ordered)
    return logits


def _sample_next(
    last_logits: torch.Tensor, *, temperature: float, top_k: int, top_p: float, generator: torch.Generator | None
) -> int:
    if temperature <= 0.0:  # greedy
        return int(last_logits.argmax(dim=-1))
    logits = _filter_logits(last_logits / temperature, top_k, top_p)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator))


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    max_context: int | None = None,
    seed: int | None = None,
) -> list[int]:
    """Autoregressively extend ``prompt_ids``, returning prompt + generated ids.

    ``temperature <= 0`` is greedy. ``max_context`` crops the conditioning window
    to the model's context length. ``seed`` makes sampling reproducible.
    """
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        context = ids[-max_context:] if max_context else ids
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x, torch.ones_like(x))  # [1, T, vocab]
        last = logits[0, -1].detach().to("cpu").float()
        nxt = _sample_next(last, temperature=temperature, top_k=top_k, top_p=top_p, generator=generator)
        ids.append(nxt)
        if eos_token_id is not None and nxt == eos_token_id:
            break
    return ids


def generate_sequence(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str = "",
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    max_context: int | None = None,
    seed: int | None = None,
) -> str:
    """Complete a sequence from a (possibly empty) prompt and decode it to bases."""
    prompt_ids = tokenizer.tokenize_content(prompt) if prompt else []
    out = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_context=max_context,
        seed=seed,
    )
    return tokenizer.decode(out)
