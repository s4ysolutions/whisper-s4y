import logging
import os
import tempfile
import tensorflow as tf

from abc import ABC, abstractmethod
from whisper_s4y import artefact_path

_def_log = logging.getLogger(os.path.basename(os.path.dirname(__file__)))


class ServingModel(tf.Module, ABC):
    @abstractmethod
    def serving(self, *args, **kwargs):
        pass


def _save(model_name: str, model: ServingModel, saved_model_dir: str = None, log: logging.Logger = _def_log) -> str:
    if saved_model_dir is None:
        saved_model_dir = os.path.join(tempfile.gettempdir(), 'whisper2tflite', model_name)

    log.debug(f"{model_name} save start...")
    tf.saved_model.save(model, saved_model_dir, signatures={"serving_default": model.serving})
    log.info(f"{model_name} save done in  {saved_model_dir}")
    return saved_model_dir


def _tflite(saved_model_dir: str, tflite_model_path: str, optimize: bool = True, log: logging.Logger = _def_log):
    name = os.path.basename(tflite_model_path)
    log.info(f"{name} create converter start...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    log.info(f"{name} create converter done")
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    else:
        converter.optimizations = []
    #converter.inference_input_type = tf.float32
    #converter.inference_output_type = tf.float32
    log.info(f"{name} converting start...")
    tflite_model = converter.convert()
    log.info(f"{name} converting done")
    log.debug(f"{name} converted model save start...")
    dir = os.path.dirname(tflite_model_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    log.info(f"{name} converted model save done: {tflite_model_path}")


def tflite(model_name: str, model: ServingModel, tflite_model_path: str = None,
           log: logging.Logger = _def_log, optimize=True, artefact=False) -> str:
    saved_model_dir = _save(model_name, model, log=log)
    if tflite_model_path is None:
        if artefact:
            tflite_model_path = artefact_path(f"{model_name}.tflite")
        else:
            tflite_model_path = f"{model_name}.tflite"
    else:
        if artefact:
            log.warning("artefact=True is ignored when tflite_model_path is provided")
    if not os.path.isabs(tflite_model_path):
        tflite_model_path = os.path.join(saved_model_dir, tflite_model_path)
    _tflite(saved_model_dir, tflite_model_path, optimize=optimize, log=log)
    return tflite_model_path


if __name__ == "__main__":
    from datetime import datetime
    import os
    import sys
    import logging

    log = logging.getLogger(os.path.basename(os.path.dirname(__file__)))
    log.setLevel(logging.DEBUG)


    # log.addHandler(logging.StreamHandler(sys.stdout))

    class InnerModel(tf.Module):
        def __call__(self, x, n=None):
            n = 2 if n is None else n
            return x * n


    class TestModel(ServingModel):
        def __init__(self):
            super().__init__()
            self.inner = InnerModel()

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=None, dtype=tf.int32, name="x"),
            ],
        )
        def serving(self, x):
            y = self.inner(x, None)
            return {"x": x, "y": y}


    tflite_path = tflite('test', TestModel(), log=log)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    runner = interpreter.get_signature_runner()
    x = tf.constant([1, 10], dtype=tf.int32)
    output = runner(x=x)
    log.debug(f"Test output: {output}")
    tf.debugging.assert_equal(output['x'], x)
    tf.debugging.assert_equal(output['y'], x * 2)

#    @tf.function
#    def my_func(x, y):
#        # A simple hand-rolled layer.
#        return tf.nn.relu(tf.matmul(x, y))
#
## Set up logging.
#    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#    logdir = 'logs/func/%s' % stamp
#    writer = tf.summary.create_file_writer(logdir)
#
#    x = tf.random.uniform((3, 3))
#    y = tf.random.uniform((3, 3))
#
#    tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=logdir)
#    z = my_func(x, y)
#
#    with writer.as_default():
#        tf.summary.trace_export(
#            name="my_func_trace",
#            step=0,
#            profiler_outdir=logdir)
