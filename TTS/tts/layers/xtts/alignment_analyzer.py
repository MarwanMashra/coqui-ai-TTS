"""alignment_analyzer.py
========================
A lightweight runtime guard for streaming **TTS** decoders
---------------------------------------------------------
The module tracks the *cross‑attention* between generated acoustic
frames (queries) and the input text tokens (keys).  From that signal it
attempts to detect two common failure modes and—when either is met—forces
an **EOS** logit so that decoding stops early instead of hallucinating
noise:

1. **Hallucination loop** – the decoder keeps re‑attending a single text
   token (empirically token index 1, often a comma) for many consecutive
   frames.  We maintain a sliding boolean window over the last
   ``WINDOW_FRAMES`` frames, compute its *loop ratio* and cut when the
   ratio exceeds ``LOOP_RATIO_CUTOFF``.
2. **Tail completed** – every text token has been *visited* according to
   an attention‑streak heuristic.  Once the final token is visited we
   allow ``EOS_HOLD_FRAMES`` more frames (for trailing silence) and then
   cut.

The implementation deliberately avoids heavyweight dependencies and
keeps all state on the Python side; it works with any PyTorch model as
long as an ``extract_attention`` callback is provided.
"""

from __future__ import annotations

import inspect
import math
from collections import deque
from collections.abc import Callable
from types import FunctionType, MethodType
from typing import Any

import torch
from torch.utils.hooks import RemovableHandle

ExtractAttention = Callable[[torch.nn.Module, tuple[Any, ...], tuple[Any, ...]], torch.Tensor]
PatchedForward = Callable[[Callable[..., Any], Any], Any]


class AlignmentAnalyzer:
    """Runtime guard that forces EOS when attention reveals a failure mode."""

    # ───────────────────────── Tunables ──────────────────────────
    # Upper‑bound (“cursor”) smoothing window
    _UB_LEFT: int = 2
    _UB_RIGHT: int = 4

    # Per‑token visit heuristic
    _BASE_STREAK: int = 3
    _JUMP_STREAK_SCALE: float = 0.5

    # Frames to wait after the final token is visited
    EOS_HOLD_FRAMES: int = 5

    # Hallucination‑loop guard (token index 1 is hard‑wired)
    WINDOW_FRAMES: int = 15
    LOOP_RATIO_CUTOFF: float = 0.70

    # logit clipping value (large negative)
    _NEG_INF: int = -(2**15)

    def __init__(
        self,
        attention_layer: torch.nn.Module,
        extract_attention: ExtractAttention,
        *,
        patched_forward: PatchedForward | None = None,
        verbose: bool = False,
    ) -> None:
        """Attach to *attention_layer* and start collecting attentions.

        Parameters
        ----------
        attention_layer:
            A layer whose *forward* returns attention matrices in its
            outputs.  The exact contract is hidden behind
            *extract_attention*.
        extract_attention:
            ``f(layer, inputs, outputs) -> Tensor`` returning the raw
            attention tensor of shape *(batch, heads, seq, seq)*.
        patched_forward:
            Optional wrapper to muck with the layer's forward pass (e.g.
            to shorten KV caches).  Leave *None* if you do not need it.
        verbose:
            When *True* prints one diagnostics line per frame.
        """
        self._layer = attention_layer
        self._extract_attention = extract_attention
        self._patched_forward = patched_forward
        self._verbose = verbose

        # will be initialised in :meth:`initialize`
        self._hook: RemovableHandle | None = None
        self._orig_forward_func: FunctionType | None = None
        self._last_attention: torch.Tensor | None = None
        self._ready = False

    def initialize(self, text_span: tuple[int, int], eos_token_id: int) -> None:
        """Reset all state before starting a new utterance."""
        start, end = text_span
        self._span: tuple[int, int] = text_span
        self._text_len: int = end - start
        self._eos_id: int = eos_token_id

        # frame counter & alignment trace (mainly for offline analysis)
        self._frame: int = 0
        self._alignment: torch.Tensor = torch.zeros(0, self._text_len)

        # per‑token state
        self._upper_bound: int = 0  # smoothed arg‑max pointer
        self._streak: list[int] = [0] * self._text_len
        self._visited: list[bool] = [False] * self._text_len
        self._edge: int = -1  # right‑most contiguous *visited* index

        # hallucination‑loop tracking
        self._loop_window: deque[bool] = deque(maxlen=self.WINDOW_FRAMES)
        self._loop_ratio: float = float("-inf")  # undefined until window full

        # EOS‑hold countdown
        self._eos_hold: int = 0

        # attach hook
        self._attach_hook()
        self._ready = True

    def reset(self) -> None:
        """Detach the hook and clear state (call after each utterance)."""
        self._ready = False
        self._detach_hook()

    @torch.no_grad()
    def step(self, logits: torch.Tensor) -> torch.Tensor:
        """Process one decoder frame and possibly force EOS."""
        if not self._ready:
            return logits

        arg_max = self._ingest_attention()
        self._update_loop_ratio(arg_max)
        self._update_upper_bound(arg_max)
        self._update_streaks(arg_max)
        self._advance_edge()
        self._update_eos_hold()

        if self._should_cut():
            logits.fill_(self._NEG_INF)
            logits[..., self._eos_id] = -self._NEG_INF  # positive large number
            if self._verbose:
                ratio_disp = f"{self._loop_ratio:.2f}" if math.isfinite(self._loop_ratio) else "n/a"
                print(f"AlignmentAnalyzer: force‑EOS | loop_ratio={ratio_disp} edge={self._edge} hold={self._eos_hold}")

        if self._verbose:
            last_hit = self._loop_window[-1] if self._loop_window else False
            ratio_disp = f"{self._loop_ratio:.2f}" if math.isfinite(self._loop_ratio) else "n/a"
            print(
                f"F{self._frame:04d} | arg={arg_max:3d} ub={self._upper_bound:3d} "
                f"edge={self._edge:3d}/{self._text_len - 1} dist={self._upper_bound - self._edge:2d} "
                f"hit={last_hit} ratio={ratio_disp} hold={self._eos_hold} cut={self._should_cut()}"
            )

        self._frame += 1
        return logits

    def _ingest_attention(self) -> int:
        """Grab the latest attention row and extend *self._alignment*."""
        start, end = self._span
        attn = self._last_attention  # (heads, seq, seq)
        row = attn[end:, start:end] if self._frame == 0 else attn[:, start:end]
        row = row.clone().cpu()
        self._alignment = torch.cat((self._alignment, row), 0)
        return int(row[-1].argmax())

    def _update_upper_bound(self, arg_max: int) -> None:
        """Update *upper_bound* while clamping spurious jumps."""
        if self._upper_bound - self._UB_LEFT <= arg_max <= self._upper_bound + self._UB_RIGHT:
            self._upper_bound = arg_max

    def _update_streaks(self, arg_max: int) -> None:
        start = self._edge + 1
        end = min(arg_max, self._upper_bound)
        for idx in range(start, end + 1):
            self._streak[idx] += 1
        for idx in range(end + 1, self._text_len):
            if not self._visited[idx]:
                self._streak[idx] = 0

    def _advance_edge(self) -> None:
        for idx in range(self._edge + 1, self._text_len):
            if self._visited[idx]:
                continue
            distance = idx - self._edge
            required = self._BASE_STREAK + math.ceil((distance - 1) * self._JUMP_STREAK_SCALE)
            if self._streak[idx] >= required:
                self._visited[idx] = True
            else:
                break
        while self._edge + 1 < self._text_len and self._visited[self._edge + 1]:
            self._edge += 1

    def _update_loop_ratio(self, arg_max: int) -> None:
        """Maintain sliding window of hits and compute *loop_ratio*."""
        self._loop_window.append(arg_max == 0 or arg_max == self._text_len - 1)
        if len(self._loop_window) == self.WINDOW_FRAMES:
            self._loop_ratio = sum(self._loop_window) / self.WINDOW_FRAMES
        else:
            self._loop_ratio = float("-inf")

    def _update_eos_hold(self) -> None:
        finished = self._edge >= self._text_len - 1
        self._eos_hold = self._eos_hold + 1 if finished else 0

    def _should_cut(self) -> bool:
        loop_ready = math.isfinite(self._loop_ratio)
        loop_fail = loop_ready and self._loop_ratio >= self.LOOP_RATIO_CUTOFF
        eos_done = self._eos_hold >= self.EOS_HOLD_FRAMES
        return loop_fail or eos_done

    def _attach_hook(self) -> None:
        """Attach a forward hook to *self._layer*."""

        def _hook(module, inputs, output):
            att = self._extract_attention(module, inputs, output)[0]  # (B, H, N, N)
            self._last_attention = att.mean(0).cpu()  # (N, N)

        self._hook = self._layer.register_forward_hook(_hook)

        # Optionally patch the layer's .forward for stream optimisation
        if self._patched_forward is None:
            return

        self._orig_forward_func = self._layer.forward.__func__
        param_names = list(inspect.signature(self._orig_forward_func).parameters)[1:]

        def _forward(module_self, *args, **kwargs):
            for name in param_names[: len(args)]:
                kwargs.pop(name, None)
            orig_bound = MethodType(self._orig_forward_func, module_self)
            return self._patched_forward(orig_bound, *args, **kwargs)

        self._layer.forward = MethodType(_forward, self._layer)

    def _detach_hook(self) -> None:
        if self._hook:
            self._hook.remove()
            self._hook = None
        if self._orig_forward_func:
            self._layer.forward = MethodType(self._orig_forward_func, self._layer)
            self._orig_forward_func = None
