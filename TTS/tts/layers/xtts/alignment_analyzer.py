"""
alignment_analyzer.py
=====================

**Purpose**

Detect when a streaming text-to-speech decoder has lost alignment with the
prompt text and force an early **EOS** token to suppress tail-hallucinations
and silence.

The heuristic fuses three independent signals:

1. **Visited-edge stall**
   *We expect forward momentum.*
   A running “edge” marks the highest prompt token that has been confirmed
   by attention.  If the arg-max attention index (`arg_max`) sits more than
   `EDGE_BACK_TOLERANCE` tokens **behind** that edge in a high percentage of
   the last `STALE_WINDOW_FRAMES` frames, generation is declared *stagnant*.

2. **“Stuck-at-start” loop**
   Many pathological runs bounce on **prompt token 1** (typically the comma
   after a greeting) and never recover.  If `arg_max` has been exactly
   `START_STUCK_IDX` (hard-wired = 1) in ≥ `START_STUCK_RATIO_CUTOFF`
   of the last `START_STUCK_WINDOW_FRAMES` frames, we cut.

3. **EOS hold**
   Once the visited edge reaches the final prompt token, we allow the model
   `EOS_HOLD_FRAMES` more frames to output trailing silence; after that we cut.

**How the visited edge advances**

* Let `cursor` be a smoothed version of `arg_max` that only updates when
  `arg_max` lies inside an acceptance window
  `[cursor − CURSOR_WINDOW_LEFT , cursor + CURSOR_WINDOW_RIGHT]`.
* Every frame each **unvisited** token *between the current edge and
  `min(cursor, arg_max)`* accrues a hit.
* A token `t` becomes *visited* once its consecutive hit count ≥

      VISIT_BASE_FRAMES + ceil((t − edge − 1) × VISIT_JUMP_SCALE).

  Longer jumps thus require proportionally longer confirmation.
* When several contiguous tokens satisfy the rule simultaneously, the edge
  can advance by more than one token in a single frame.

Set `verbose=True` in the constructor and enable the module logger at
DEBUG level to see frame-by-frame traces.  All thresholds are module-level
constants and can be tuned empirically.
"""

from __future__ import annotations

import inspect
import math
from collections import deque
from collections.abc import Callable
from types import FunctionType, MethodType

import torch
from torch.utils.hooks import RemovableHandle


class AlignmentAnalyzer:
    # ───────────────────────── Tunables ──────────────────────────
    CURSOR_WINDOW_LEFT = 2
    CURSOR_WINDOW_RIGHT = 4

    VISIT_BASE_FRAMES = 3
    VISIT_JUMP_SCALE = 0.5

    EDGE_BACK_TOLERANCE = 0

    STALE_WINDOW_FRAMES = 15
    STALE_RATIO_CUTOFF = 0.80
    STALE_MIN_FILL = STALE_WINDOW_FRAMES // 2

    EOS_HOLD_FRAMES = 4

    # “stuck at the very beginning” guard (token index 1 is hard-wired)
    START_STUCK_IDX = 1
    START_STUCK_WINDOW_FRAMES = 10
    START_STUCK_RATIO_CUTOFF = 0.80
    # ------------------------------------------------------------

    def __init__(
        self,
        attention_layer: torch.nn.Module,
        extract_attention: Callable[[torch.nn.Module, tuple, tuple], torch.Tensor],
        *,
        patched_forward: Callable | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        attention_layer
            Decoder self-attention layer to observe.
        extract_attention
            Callback returning (B, heads, N, N) attention given
            (module, inputs, output).
        force_output_attention
            Patch the target layer so it always emits its attention map.
        verbose
            If `True`, print frame-by-frame debug traces to the console.
        """
        self._layer = attention_layer
        self._extract_attention = extract_attention
        self._patched_forward = patched_forward
        self._verbose = verbose

        # runtime state filled by `initialize`
        self._hook: RemovableHandle | None = None
        self._orig_forward_func: FunctionType | None = None
        self._last_attention: torch.Tensor | None = None
        self._ready = False

    # ─────────────────────── Public API ──────────────────────────

    def initialize(self, text_span: tuple[int, int], eos_token_id: int) -> None:
        """
        Initialize internal state to get ready for a new generation pass.

        Parameters
        ----------
        text_span
            (start, end) slice of prompt tokens inside model sequence.
        eos_token_id
            Vocabulary id for EOS token.
        """
        start, end = text_span
        self._text_len = end - start
        self._span = text_span
        self._eos_id = eos_token_id

        # counters & trackers
        self._frame = 0
        self._alignment = torch.zeros(0, self._text_len)

        self._cursor = 0
        self._streak = [0] * self._text_len
        self._visited = [False] * self._text_len
        self._edge = -1

        self._stale_hist: deque[bool] = deque(maxlen=self.STALE_WINDOW_FRAMES)
        self._stuck_hist: deque[bool] = deque(maxlen=self.START_STUCK_WINDOW_FRAMES)
        self._eos_hold = 0

        self._attach_hook()
        self._ready = True

    def reset(self) -> None:
        """Detach the attention hook and clear state."""
        self._detach_hook()
        self._ready = False

    # ───────────────────────── Main step ──────────────────────────

    @torch.no_grad()
    def step(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._ready:
            return logits

        arg_max = self._ingest_attention()
        self._update_cursor(arg_max)
        self._update_streaks(arg_max)
        self._advance_edge()
        self._update_stale(arg_max)
        self._update_stuck(arg_max)
        self._update_eos_hold()

        if self._should_cut():
            logits.fill_(-(2**15))
            logits[..., self._eos_id] = 2**15
            if self._verbose:
                print(
                    f"AlignmentAnalyzer: force EOS | edge={self._edge} stale={self._stale_ratio:.2f} "
                    f"stuck={self._stuck_ratio:.2f} eos={self._eos_hold}"
                )

        if self._verbose:
            print(
                f"AlignmentAnalyzer: F{self._frame:04d} | arg={arg_max:3d} cur={self._cursor:3d} edge={self._edge:3d} "
                f"dist={self._cursor - self._edge:2d} stale={self._stale_hist[-1] if self._stale_hist else False}({self._stale_ratio:.2f}) "
                f"stuck={self._stuck_hist[-1] if self._stuck_hist else False}({self._stuck_ratio:.2f}) "
                f"eos={self._eos_hold} cut={self._should_cut()}"
            )

        self._frame += 1
        return logits

    # ─────────────────── Internal mechanics ──────────────────────

    # Attention ingestion ----------------------------------------------------
    def _ingest_attention(self) -> int:
        start, end = self._span
        attn = self._last_attention
        row = attn[end:, start:end] if self._frame == 0 else attn[:, start:end]
        row = row.clone().cpu()
        row[:, 0] = 0  # mask BOS
        self._alignment = torch.cat((self._alignment, row), 0)
        return int(row[-1].argmax())

    # Cursor smoothing ------------------------------------------------------
    def _update_cursor(self, arg_max: int) -> None:
        """
        Move the smoothing cursor.

        * Follow the arg-max if it lies within the acceptance window
        [cursor−L , cursor+R] (usual case).
        * If the arg-max lies **outside** that range, re-center the cursor
        immediately.  This prevents the “stuck-since-frame-0” bug.
        """
        if arg_max < self._cursor - self.CURSOR_WINDOW_LEFT or arg_max > self._cursor + self.CURSOR_WINDOW_RIGHT:
            self._cursor = arg_max
        elif (
            self._frame == 0
            or self._cursor - self.CURSOR_WINDOW_LEFT <= arg_max <= self._cursor + self.CURSOR_WINDOW_RIGHT
        ):
            self._cursor = arg_max

    # Per-token consecutive streaks -----------------------------------------
    def _update_streaks(self, arg_max: int) -> None:
        start = self._edge + 1
        end = min(arg_max, self._cursor)
        for idx in range(start, end + 1):
            self._streak[idx] += 1
        for idx in range(end + 1, self._text_len):
            if not self._visited[idx]:
                self._streak[idx] = 0

    def _advance_edge(self) -> None:
        idx = self._edge + 1
        while idx < self._text_len:
            if self._visited[idx]:
                idx += 1
                continue
            distance = idx - self._edge
            required = self.VISIT_BASE_FRAMES + math.ceil((distance - 1) * self.VISIT_JUMP_SCALE)
            if self._streak[idx] >= required:
                self._visited[idx] = True
                idx += 1
            else:
                break
        while self._edge + 1 < self._text_len and self._visited[self._edge + 1]:
            self._edge += 1

    # Staleness detection ----------------------------------------------------
    def _update_stale(self, arg_max: int) -> None:
        stale = arg_max < self._edge - self.EDGE_BACK_TOLERANCE
        self._stale_hist.append(stale)
        self._stale_ratio = sum(self._stale_hist) / len(self._stale_hist)

    # “Stuck at start” detection --------------------------------------------
    def _update_stuck(self, arg_max: int) -> None:
        self._stuck_hist.append(arg_max == self.START_STUCK_IDX)
        self._stuck_ratio = sum(self._stuck_hist) / len(self._stuck_hist)

    # EOS hold ---------------------------------------------------------------
    def _update_eos_hold(self) -> None:
        at_end = self._edge >= self._text_len - 1
        self._eos_hold = self._eos_hold + 1 if at_end else 0

    # Final cut decision -----------------------------------------------------
    def _should_cut(self) -> bool:
        # NOTE: stagnant condition is dangerous, maybe remove it?
        # stagnant = len(self._stale_hist) >= self.STALE_MIN_FILL and self._stale_ratio >= self.STALE_RATIO_CUTOFF
        stagnant = False
        stuck_loop = (
            len(self._stuck_hist) == self.START_STUCK_WINDOW_FRAMES
            and self._stuck_ratio >= self.START_STUCK_RATIO_CUTOFF
        )
        eos_done = self._eos_hold >= self.EOS_HOLD_FRAMES
        return stagnant or stuck_loop or eos_done

    # Hook plumbing ----------------------------------------------------------
    def _attach_hook(self) -> None:
        """Grab attention and (optionally) patch the layer's .forward."""

        # 1. Forward-hook to store the latest attention
        def _hook(module, inputs, output):
            att = self._extract_attention(module, inputs, output)[0]  # (B,H,N,N)
            self._last_attention = att.mean(0).cpu()  # (N,N)

        self._hook = self._layer.register_forward_hook(_hook)

        # 2. Patch the forward pass
        if self._patched_forward is None:
            return

        self._orig_forward_func = self._layer.forward.__func__
        param_names = list(inspect.signature(self._orig_forward_func).parameters)[1:]

        def _forward(module_self, *args, **kwargs):
            """
            Bound shim that:
            • drops keyword duplicates already given positionally
            • wraps the original forward with user-provided patched_forward
            """
            for name in param_names[: len(args)]:
                kwargs.pop(name, None)  # deletes layer_past / attention_mask duplicates

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
