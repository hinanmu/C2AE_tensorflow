#@Time      :2019/3/26 16:39
#@Author    :zhounan
# @FileName: optimizers.py
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf

class CustomOptimizer(optimizer.Optimizer):

    def __init__(self, learning_rate, decay=0.9, momentum=0.99, use_locking=False, name="CustomOptimizer"):
        super(CustomOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._decay = decay
        self._momentum = momentum

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._decay_t = None
        self._momentum_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._decay_t = ops.convert_to_tensor(self._decay, name="decay_t")
        self._momentum_t = ops.convert_to_tensor(self._momentum, name="momentum_t")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "rms", self._name)
            self._zeros_slot(v, "mom", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        decay_t = math_ops.cast(self._decay_t, var.dtype.base_dtype)
        momentum_t = math_ops.cast(self._momentum_t, var.dtype.base_dtype)

        rms = self.get_slot(var, "rms")
        mom = self.get_slot(var, "mom")
        rms_t = state_ops.assign(rms, math_ops.sqrt(decay_t * rms * rms + (1-decay_t) * grad * grad))
        mom_t = state_ops.assign(mom, momentum_t * mom - lr_t * grad / rms_t)

        var_update = state_ops.assign_add(var, mom_t)
        return control_flow_ops.group(*[var_update, rms_t, mom_t])

    def _apply_sparse(self, grad, var):
        return NotImplementedError("Sparse gradient updates are not supported yet.")
