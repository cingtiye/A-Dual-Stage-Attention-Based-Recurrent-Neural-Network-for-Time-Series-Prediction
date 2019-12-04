# ==============================================================================
# Author: Feiteng Li
# ==============================================================================

"""LookAhead for TensorFlow."""
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


class LookaheadOptimizer(optimizer.Optimizer):
  """Wrapper optimizer that implements the Lookahead Optimizer.

  See [Michael et al., 2019](https://arxiv.org/abs/1907.08610)
  ([pdf](https://arxiv.org/pdf/1907.08610.pdf)).
  """

  def __init__(self,
               opt,
               interval_steps=5,  # k
               alpha=0.5,
               name="Lookahead"):
    """Construct a new model average optimizer.

    Args:
      opt: The actual optimizer that will be used to update local variables
      interval_steps: An int point value to controls the frequency of the
        update of slow variables
      name: string. Optional name of the returned operation
    """
    super(LookaheadOptimizer, self).__init__(opt._use_locking, name)
    self._opt = opt
    self._interval_steps = interval_steps
    self._alpha = alpha

  def _get_step_accumulator(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return self._get_non_slot_variable("step", graph=graph)

  def compute_gradients(self, *args, **kwargs):
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the compute_gradients() from the real optimizer.

    Args:
      *args: Arguments for compute_gradients().
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    """
    return self._opt.compute_gradients(*args, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    This contains most of the synchronization implementation and also wraps the
    apply_gradients() from the real optimizer. The chief work updates global
    variables.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      A conditional 'Operation' that update both local and global variables or
      just local variables

    Raises:
      ValueError: If the grads_and_vars is empty.
      ValueError: If global step is not provided, the staleness cannot be
        checked.
    """
    if not grads_and_vars:
      raise ValueError("Must supply at least one variable")
    if global_step is None:
      print("{} WARNING: Global step is None.".format(self._name))

    var_list = [v for g, v in grads_and_vars if g is not None]
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(initial_value=0,
                                   name="step",
                                   colocate_with=first_var)
    # Create slots for local vars.
    for var in var_list:
      self._zeros_slot(var, 'local', self._name)

    apply_updates = self._opt.apply_gradients(grads_and_vars, global_step=global_step)
    with ops.control_dependencies([apply_updates]):
      local_update = state_ops.assign_add(self._get_step_accumulator(), 1, name="local_step_update").op

    def _update_variables():
      global_assignments = []
      local_assignments = []
      for (grad, var) in grads_and_vars:
        if grad is None or var is None:
          continue

        local = self.get_slot(var, 'local')

        next_var = (1.0 - self._alpha) * local + self._alpha * var
        global_assignments.append(state_ops.assign(var, next_var, use_locking=self._opt._use_locking))
        local_assignments.append(state_ops.assign(local, next_var, use_locking=self._opt._use_locking))

      with ops.control_dependencies(global_assignments):
        # update local variables.
        return control_flow_ops.group(*(local_assignments))

    with ops.control_dependencies([local_update]):
      condition = math_ops.equal(math_ops.mod(self._get_step_accumulator(), self._interval_steps), 0)
      conditional_update = control_flow_ops.cond(condition,
                                                 true_fn=_update_variables,
                                                 false_fn=control_flow_ops.no_op)

    with ops.control_dependencies([conditional_update]):
      return control_flow_ops.no_op()
