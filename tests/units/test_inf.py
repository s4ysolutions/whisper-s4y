import math
import tensorflow as tf


def test_inf_can_not_be_added():
    assert math.isinf(float("-inf") + 1)
    assert math.isinf(float("inf") + float("-inf"))


def test_inf_can_be_compared():
    assert float("-inf") < 0
    minus_inf_plus_1 = float("-inf") + 1
    assert not float("-inf") < minus_inf_plus_1


def test_floats_can_be_compared():
    min_float = tf.constant(-3.4028235e+38, tf.float32)
    min_float_plus1 = tf.constant(-3.4028235e+38 + 1, tf.float32)
    assert not min_float < min_float_plus1
    min_float_significant = tf.constant(-3.4028236e+38, tf.float32)
    assert math.isinf(min_float_significant)
