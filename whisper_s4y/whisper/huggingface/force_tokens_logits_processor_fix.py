import numpy as np

from typing import List
from transformers import TFForceTokensLogitsProcessor
import tensorflow as tf


# Patching methods of class TFForceTokensLogitsProcessor(TFLogitsProcessor):
# TFForceTokensLogitsProcessor has a bug which causes lite model to crash
# to fix it, the 2 methods are overridden and replaced
# https://github.com/huggingface/transformers/issues/19691#issuecomment-1791869884

def patched__init__(self, force_token_map: List[List[int]]):
    force_token_map = dict(force_token_map)
    # Converts the dictionary of format {index: token} containing the tokens to be forced to an array, where the
    # index of the array corresponds to the index of the token to be forced, for XLA compatibility.
    # Indexes without forced tokens will have an negative value.
    force_token_array = np.ones((max(force_token_map.keys()) + 1), dtype=np.int32) * -1
    for index, token in force_token_map.items():
        if token is not None:
            force_token_array[index] = token
    self.force_token_array = tf.convert_to_tensor(force_token_array, dtype=tf.int32)


TFForceTokensLogitsProcessor.__init__ = patched__init__


def patched__call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
    def _force_token(generation_idx):
        batch_size = scores.shape[0]
        current_token = self.force_token_array[generation_idx]

        # Original code below generates NaN values when the model is exported to tflite
        # it just needs to be a negative number so that the forced token's value of 0 is the largest
        # so it will get chosen
        # new_scores = tf.ones_like(scores, dtype=scores.dtype) * -float("inf")
        new_scores = tf.ones_like(scores, dtype=scores.dtype) * -float(1)
        indices = tf.stack((tf.range(batch_size), tf.tile([current_token], [batch_size])), axis=1)
        updates = tf.zeros((batch_size,), dtype=scores.dtype)
        new_scores = tf.tensor_scatter_nd_update(new_scores, indices, updates)
        return new_scores

    scores = tf.cond(
        tf.greater_equal(cur_len, tf.shape(self.force_token_array)[0]),
        # If the current length is geq than the length of force_token_array, the processor does nothing.
        lambda: tf.identity(scores),
        # Otherwise, it may force a certain token.
        lambda: tf.cond(
            tf.greater_equal(self.force_token_array[cur_len], 0),
            # Only valid (positive) tokens are forced
            lambda: _force_token(cur_len),
            # Otherwise, the processor does nothing.
            lambda: scores,
        ),
    )
    return scores


TFForceTokensLogitsProcessor.__call__ = patched__call__
