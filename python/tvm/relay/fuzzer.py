import types
import inspect
import functools
import random

import tvm
from tvm import relay
from tvm.relay.testing import mlp
from tvm.relay.testing import check_grad
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

    if not check_outs(out_org, out_new):
        return (conc_args, out_org, out_new)

# Functions for fuzzing
def sqrt(expr):
    return relay.sqrt(relay.abs(expr))

def rsqrt(expr):
    return relay.rsqrt(relay.add(relay.abs(expr), relay.const(1.0)))

def log(expr):
    return relay.log(relay.add(relay.abs(expr), relay.const(1.0)))

def div(top, bot):
    return relay.div(top, relay.add(relay.abs(bot), relay.const(1.0)))

def mod(top, bot):
    return relay.mod(top, relay.add(relay.abs(bot), relay.const(1.0)))

def conv2d(e1, e2):
    x1 = relay.reshape(e1, (2, 2, 2, 2))
    x2 = relay.reshape(e2, (2, 2, 2, 2))
    return relay.nn.conv2d(x1, x2)


OPS = [
        {'func': relay.add,      'arity': 2, 'weight': 1},
        {'func': relay.subtract, 'arity': 2, 'weight': 1},
        {'func': relay.multiply, 'arity': 2, 'weight': 1},
        {'func': div,   'arity': 2, 'weight': 0},
        {'func': relay.sign,     'arity': 1, 'weight': 1},
        {'func': relay.sigmoid,  'arity': 1, 'weight': 1},
        {'func': relay.tanh,     'arity': 1, 'weight': 1},
        {'func': relay.nn.relu,  'arity': 1, 'weight': 1},
        {'func': log,      'arity': 1, 'weight': 1},
        {'func': sqrt,     'arity': 1, 'weight': 1},
        {'func': rsqrt,    'arity': 1, 'weight': 1},
        {'func': relay.exp,      'arity': 1, 'weight': 1},
        {'func': mod,      'arity': 2, 'weight': 1},
        {'func': conv2d,  'arity': 2, 'weight': 1},
        {'func': relay.nn.dense,     'arity': 2, 'weight': 1},

]

def choose_op():
    weights = []
    for op in OPS:
        weights.append(op['weight'])
    return random.choices(OPS, weights=weights)[0]

SHAPE = 4, 4

def gen_op_seq(ops, length):
    seq = []
    for i in range(length):
        seq.append(choose_op())

    return seq

def gen_prog(op_seq, shape, dtype):
    init = relay.var('arg', shape=shape, dtype=dtype)
    lives = [init]
    out = init

    for op in op_seq:
        # first arg is always the last computed var
        args = [out]
        # Remaining arguments are randomly chosen from live set
        for i in range(op['arity'] - 1):
            args.append(random.choice(lives))

        # output var
        out = op['func'](*args)
        lives.append(out)
    args = relay.analysis.free_vars(out)
    return relay.Function(args, out)

def fuzz_pass(tvm_pass,
              prog_len=10,
              ops=OPS,
              shape=SHAPE,
              dtype='float32',
              runs=300):

    for i in range(runs):
        prog = gen_prog(gen_op_seq(ops, prog_len), shape, dtype)
        res = fuzz_expr(prog, tvm_pass)
        assert res is None, "Pass get different outputs on:\n{},\nouts:{}".format(prog, res)
        # Bug we found!
        # check_grad(prog)


def test():
    tvm_pass = transform.PartialEvaluate()
    fuzz_pass(tvm_pass)

def failing_tvm_typecheck():
    tvm_pass = transform.PartialEvaluate()
    ops = [
        {'func': relay.nn.conv2d,  'arity': 2, 'weight': 1},
    ]
    fuzz_pass(tvm_pass, prog_len=2, ops = ops, shape=[1, 1, 2, 2])

if __name__ == '__main__':
    SEED = 5
    np.random.seed(SEED)

    print('========== STARTING ============')
    print('Testing ...')
    print('\tSEED:%d'%SEED)
    test()
    # failing_tvm_typecheck()
    print('========== SUCCESS! ============')
