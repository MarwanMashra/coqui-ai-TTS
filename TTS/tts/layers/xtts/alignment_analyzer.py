from collections.abc import Callable
from types import MethodType

import torch
from torch.utils.hooks import RemovableHandle


class AlignmentAnalyzer:
    def __init__(
        self,
        alignment_layer: torch.nn.Module,
        text_tokens_slice: tuple[int, int],
        forward_output_to_attn_weights: Callable[[tuple], torch.Tensor],
        eos_idx: int,
    ):
        """
        Some transformer TTS models implicitly solve text-speech alignment in one or more of their self-attention
        activation maps. This module exploits this to perform online integrity checks which streaming.
        A hook is injected into the specified attention layer, and heuristics are used to determine alignment
        position, repetition, etc.

        NOTE: currently requires no queues.
        """
        # self.queue = queue
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx
        self.alignment = torch.zeros(0, j - i)
        # self.alignment_bin = torch.zeros(0, j-i)
        self.curr_frame_pos = 0
        self.text_position = 0

        self.started = False
        self.started_at = None

        self.complete = False
        self.completed_at = None

        # Using `output_attentions=True` is incompatible with optimized attention kernels, so
        # using it for all layers slows things down too much. We can apply it to just one layer
        # by intercepting the kwargs and adding a forward hook (credit: jrm)
        self.last_aligned_attn = None
        self.alignment_layer = alignment_layer
        self._add_attention_spy()
        self.forward_output_to_attn_weights = forward_output_to_attn_weights
        self.hit_end_couter = 0
        self.did_hit_end = False
        self.hook_handle: RemovableHandle | None = None
        self.original_forward: Callable | None = None

    def unhook(self):
        """
        Unhooks the attention layer to stop collecting outputs and restore the original forward method.
        """
        print("Unhooking AlignmentAnalyzer...")
        # NOTE: this doesn't work, both are None
        if self.hook_handle is not None:
            print("Removing hook from alignment layer")
            self.hook_handle.remove()
            self.hook_handle = None

        # Restore original forward method
        if self.original_forward is not None:
            print("Restoring original forward method of alignment layer")
            self.alignment_layer.forward = MethodType(self.original_forward, self.alignment_layer)
            self.original_forward = None
            print("self.alignment_layer.forward:", self.alignment_layer.forward)

    def _add_attention_spy(self):
        """
        Adds a forward hook to a specific attention layer to collect outputs.
        Using `output_attentions=True` is incompatible with optimized attention kernels, so
        using it for all layers slows things down too much.
        (credit: jrm)
        """

        def attention_forward_hook(module, input, output):
            """
            See `LlamaAttention.forward`; the output is a 3-tuple: `attn_output, attn_weights, past_key_value`.
            NOTE:
            - When `output_attentions=True`, `LlamaSdpaAttention.forward` calls `LlamaAttention.forward`.
            - `attn_output` has shape [B, H, T0, T0] for the 0th entry, and [B, H, 1, T0+i] for the rest i-th.

            DeepSpeedSelfAttention: forward() -> [output, key_layer, value_layer, context_layer, inp_norm]
            """
            print(len(output))
            print(output[0].shape)
            print(output[1].shape)
            print(output[2].shape)
            print(output[3].shape)
            print(output[4].shape)
            step_attention = self.forward_output_to_attn_weights(output).cpu()  # (B, 16, N, N)
            self.last_aligned_attn = step_attention[0].mean(0)  # (N, N)

        self.hook_handle = self.alignment_layer.register_forward_hook(attention_forward_hook)

        # Backup original forward
        original_forward = self.alignment_layer.forward

        def patched_forward(_, *args, **kwargs):
            # kwargs["output_attentions"] = True
            return original_forward(*args, **kwargs)

        # TODO: how to unpatch it?
        self.alignment_layer.forward = MethodType(patched_forward, self.alignment_layer)
        self.original_forward = original_forward

    def step(self, logits):
        """
        Emits an AlignmentAnalysisResult into the output queue, and potentially modifies the logits to force an EOS.
        """
        # extract approximate alignment matrix chunk (1 frame at a time after the first chunk)
        aligned_attn = self.last_aligned_attn  # (N, N)
        i, j = self.text_tokens_slice
        if self.curr_frame_pos == 0:
            # first chunk has conditioning info, text tokens, and BOS token
            A_chunk = aligned_attn[j:, i:j].clone().cpu()  # (T, S)
        else:
            # subsequent chunks have 1 frame due to KV-caching
            A_chunk = aligned_attn[:, i:j].clone().cpu()  # (1, S)
        A_chunk[:, 0] = 0

        self.alignment = torch.cat((self.alignment, A_chunk), dim=0)

        A = self.alignment
        S, T = A.shape  # Q: so T is speech tokens, S is text tokens? weird naming...

        # update position
        # print(A_chunk[-1])
        cur_text_posn = A_chunk[-1].argmax()
        continuity = -4 < cur_text_posn - self.text_position < 7  # NOTE: very lenient!
        if continuity:
            self.text_position = cur_text_posn

        if self.text_position == T - 1:
            self.hit_end_couter += 1
            if self.hit_end_couter > 3:
                self.did_hit_end = True

        print(
            f"AlignmentAnalyzer: {self.curr_frame_pos=}, cur_text_posn={cur_text_posn}, {self.text_position=}, text_len={T}"
        )

        if self.did_hit_end and cur_text_posn < T - 1:
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15
            print("AlignmentAnalyzer: forcing EOS due to did_hit_end")

        self.curr_frame_pos += 1

        return logits
