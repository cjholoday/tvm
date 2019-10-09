import types
import inspect
import functools
import random

import tvm
from tvm import relay
from tvm.relay.testing import mlp
from tvm.relay import transform
from tvm.relay.backend import interpreter

import numpy as np

def str_to_np_type(ty):
    if ty == 'float32':
        return np.float32
    elif ty == 'float64':
        return np.float64
    else:
        assert False, "Cannot generate random value for type {}".format(ty)

def gen_tensor_random(shape, dtype):
    return interpreter.TensorValue(np.random.rand(*shape).astype(dtype))

def gen_random_type(ty):
    if isinstance(ty, relay.TensorType):
        return gen_tensor_random(ty.concrete_shape, str_to_np_type(ty.dtype))
    else:
        assert False, "Cannot generate random value for type {}".format(ty)

def check_outs(o1, o2):
    return np.array_equiv(o1.asnumpy(), o2.asnumpy())

def fuzz_expr(expr, tvm_pass):
    mod = tvm.relay.Module()
    mod["main"] = expr

    # Generate random concrete arguments based on the argument types
    conc_args = []
    for arg_type in mod["main"].checked_type.arg_types:
        conc_args.append(gen_random_type(arg_type))

    # Get the output of running the expr on concrete inputs
    intrp = relay.create_executor()
    out_org = intrp.evaluate(expr)(*conc_args)

    # Run the pass
    new_mod = tvm_pass(mod)
    intrp = relay.create_executor()
    out_new = intrp.evaluate(new_mod["main"])(*conc_args)

    assert check_outs(out_org, out_new), "Outputs differ.\nout_org: {}\nout_new: {}".format(out_org, out_new)

# tvm_pass = transform.PartialEvaluate()
print(mlp.get_net(1))
# for i in range(10):
    # fuzz_expr(mlp.get_net(1), tvm_pass)

OPS = relay.add, relay.subtract, relay.multiply, relay.divide, relay.sign

OPS = [
      {'func': relay.add,      'arity': 2},
      {'func': relay.subtract, 'arity': 2},
      {'func': relay.multiply, 'arity': 2},
      {'func': relay.divide,   'arity': 2},
      {'func': relay.sign,     'arity': 2},
]
SHAPE = 1, 1, 28, 28

def gen_op_seq(ops, length):
    seq = []
    for i in range(length):
        seq.append(random.choice(ops))

    return seq

x = relay.var("x")
print(relay.add(x, x))
print(gen_op_seq(OPS, 3))

print('=============================')
