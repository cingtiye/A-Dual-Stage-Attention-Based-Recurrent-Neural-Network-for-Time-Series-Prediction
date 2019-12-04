# ==============================================================================
# Author: Feiteng Li
# ==============================================================================

"""RAdam for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export


@tf_export("train.RAdamOptimizer")
class RAdamOptimizer(optimizer.Optimizer):
  """Optimizer that implements the RAdam algorithm.

  See [Liyuan Liu et al., 2019](https://arxiv.org/abs/1908.03265)
  ([pdf](https://arxiv.org/pdf/1908.03265.pdf)).
  """

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
               use_locking=False, name="RAdam"):
    """Construct a new Rectified Adam optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta1: A float value or a constant float tensor.
        The exponential decay rate for the 1st moment estimates.
      beta2: A float value or a constant float tensor.
        The exponential decay rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "RAdam".

    @compatibility(eager)
    When eager execution is enabled, `learning_rate`, `beta1`, `beta2`, and
    `epsilon` can each be a callable that takes no arguments and returns the
    actual value to use. This can be useful for changing these values across
    different invocations of optimizer functions.
    @end_compatibility
    """
    super(RAdamOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

    # Created in SparseApply if needed.
    self._updated_lr = None

  def _get_beta_accumulators(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("step", graph=graph),
              self._get_non_slot_variable("beta1_power", graph=graph),
              self._get_non_slot_variable("beta2_power", graph=graph))

  def _create_slots(self, var_list):
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(initial_value=1.0,
                                   name="step",
                                   colocate_with=first_var)
    self._create_non_slot_variable(initial_value=self._beta1,
                                   name="beta1_power",
                                   colocate_with=first_var)
    self._create_non_slot_variable(initial_value=self._beta2,
                                   name="beta2_power",
                                   colocate_with=first_var)

    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _prepare(self):
    lr = self._call_if_callable(self._lr)
    beta1 = self._call_if_callable(self._beta1)
    beta2 = self._call_if_callable(self._beta2)
    epsilon = self._call_if_callable(self._epsilon)

    self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
    self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
    self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
    self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

  def _apply_dense_shared(self, grad, var, m, v):
    step, beta1_power, beta2_power = self._get_beta_accumulators()
    step = math_ops.cast(step, var.dtype.base_dtype)
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

    # m_t = beta1 * m + (1 - beta1) * g_t
    m_t = state_ops.assign(m, m * beta1_t + grad * (1.0 - beta1_t),
                           use_locking=self._use_locking)

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v_t = state_ops.assign(v, v * beta2_t + (grad * grad) * (1.0 - beta2_t),
                           use_locking=self._use_locking)

    rho_inf = math_ops.cast(2.0 / (1.0 - self._beta2) - 1.0, var.dtype.base_dtype)
    rho_t = rho_inf - step * (2.0 * beta2_power / (1.0 - beta2_power))

    r_t = math_ops.sqrt(
        (1.0 - beta2_power) * ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf) / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))

    update = control_flow_ops.cond(math_ops.greater(rho_t, 5.0),
                                   true_fn=lambda: (lr_t / (1.0 - beta1_power) * r_t) * (
                                       m_t / (math_ops.sqrt(v_t) + epsilon_t)),
                                   false_fn=lambda: (lr_t / (1.0 - beta1_power)) * m_t)

    var_update = state_ops.assign_sub(var, update,
                                      use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def _apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    return self._apply_dense_shared(grad, var, m, v)

  def _resource_apply_dense(self, grad, var):
    return self._apply_dense(grad, var)

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    step, beta1_power, beta2_power = self._get_beta_accumulators()
    step = math_ops.cast(step, var.dtype.base_dtype)
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - beta1_t)
    m_t = state_ops.assign(m, m * beta1_t,
                           use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = scatter_add(m, indices, m_scaled_g_values)
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1 - beta2_t)
    v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = scatter_add(v, indices, v_scaled_g_values)

    rho_inf = math_ops.cast(2.0 / (1.0 - self._beta2) - 1.0, var.dtype.base_dtype)
    rho_t = rho_inf - step * (2.0 * beta2_power / (1.0 - beta2_power))

    r_t = math_ops.sqrt(
        (1.0 - beta2_power) * ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf) / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))

    update = control_flow_ops.cond(math_ops.greater(rho_t, 5.0),
                                   true_fn=lambda: (lr_t / (1.0 - beta1_power) * r_t) * (
                                       m_t / (math_ops.sqrt(v_t) + epsilon_t)),
                                   false_fn=lambda: (lr_t / (1.0 - beta1_power)) * m_t)

    var_update = state_ops.assign_sub(var, update,
                                      use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def _apply_sparse(self, grad, var):
    return self._apply_sparse_shared(
        grad.values, var, grad.indices,
        lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
            x, i, v, use_locking=self._use_locking))

  def _resource_scatter_add(self, x, i, v):
    with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_add(
            x.handle, i, v)]):
      return x.value()

  def _resource_apply_sparse(self, grad, var, indices):
    return self._apply_sparse_shared(
        grad, var, indices, self._resource_scatter_add)

  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      step, beta1_power, beta2_power = self._get_beta_accumulators()
      with ops.colocate_with(beta1_power):
        update_step = step.assign(
            step + 1.0, use_locking=self._use_locking)
        update_beta1 = beta1_power.assign(
            beta1_power * self._beta1_t, use_locking=self._use_locking)
        update_beta2 = beta2_power.assign(
            beta2_power * self._beta2_t, use_locking=self._use_locking)
    return control_flow_ops.group(*update_ops + [update_step, update_beta1, update_beta2],
                                  name=name_scope)
