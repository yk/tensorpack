from .base import Trainer
from ..utils import logger
import tensorflow as tf


class GPUMixin(Trainer):
    def _setup(self):
        ctx = None
        try:
            import pycuda.driver as pd
            import pycuda.tools as pt
            pd.init()
            ctx = pt.make_default_context()
        except:
            pass

        try:
            super()._setup()
        finally:
            if ctx is not None:
                ctx.detach()


class NoGraphSummaryMixin(Trainer):
    def _create_summary_writer(self):
        if not hasattr(logger, 'LOG_DIR'):
            raise RuntimeError(
                "Please use logger.set_logger_dir at the beginning of your script.")
        return tf.train.SummaryWriter(logger.LOG_DIR, graph=None)

