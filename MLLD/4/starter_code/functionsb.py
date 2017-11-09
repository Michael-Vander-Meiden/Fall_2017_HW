# some useful functions
import numpy as np
from xman import *
import pdb

# some useful functions
# declare all operations here first

class f(XManFunctions):
    @staticmethod
    def square(a):
        return XManFunctions.registerDefinedByOperator('square',a)
    
    @staticmethod
    def relu(a):
        return XManFunctions.registerDefinedByOperator('relu',a)

    @staticmethod
    def crossEnt(x1, x2):
        return XManFunctions.registerDefinedByOperator('crossEnt',x1,x2)

    @staticmethod
    def softMax(a):
        return XManFunctions.registerDefinedByOperator('softMax',a)

    @staticmethod
    def mean(a):
        return XManFunctions.registerDefinedByOperator('mean', a)
    
def _softMax(x):
    #x = x - np.max(x, axis=1, keepdims=True)
    #x = np.exp(x)
    #x = x / np.sum(x, axis=1, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x

    # I added other operation registers above

# the functions that autograd.eval will use to evaluate each function,
# to be called with the functions actual inputs as arguments

EVAL_FUNS = {
    'add':      lambda x1,x2: x1+x2,
    'subtract': lambda x1,x2: x1-x2,
    'square':   np.square,
    'mul':      np.dot,
    'relu':     lambda x: (x>0)*x,
    'softMax':  lambda x: np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True),
    'crossEnt': lambda x, y: -1.0*np.sum((y * np.log(x)))/y.shape[0],
    'mean':     np.mean
    }

# the functions that autograd.bprop will use in reverse mode
# differentiation.  BP_FUNS[f] is a list of functions df1,....,dfk
# where dfi is used in propagating errors to the i-th input xi of f.
# Specifically, dfi is called with the ordinary inputs to f, with two
# additions: the incoming error, and the output of the function, which
# was computed by autograd.eval in the eval stage.  dfi will return
# delta * df/dxi [f(x1,...,xk)]
#
# NOTE: Autograd has an optimization where if it finds a softMax op
# followed by crossEnt op, it combines the backward pass for both. So
# you only need to implement the BP_FUNS for the combined operation
# crossEnt-softMax below.

def _derivAdd(delta,x1):
    if delta.shape!=x1.shape:
        # broadcast, sum along axis=0
        if delta.shape[1]!=x1.shape[0]:
            raise ValueError("Dimension Mismatch")
        return delta.sum(axis=0) #we sum the gradients over the batch
    else: return delta

BP_FUNS = {
    'add':              [lambda delta,out,x1,x2: _derivAdd(delta,x1),    lambda delta,out,x1,x2: _derivAdd(delta,x2)],
    'subtract':         [lambda delta,out,x1,x2: _derivAdd(delta,x1),    lambda delta,out,x1,x2: -_derivAdd(delta,x2)],
    'square':           [lambda delta,out,x : delta * 2.0 * x],
    'mul':              [lambda delta,out,x1,x2: np.dot(delta, x2.T), lambda delta,out,x1,x2:np.dot(x1.T, delta)],
    'relu':             [lambda delta,out,x: delta * (x>0)],
    'crossEnt-softMax': [lambda delta,out,o,y: delta * (_softMax(o)-y) / y.shape[0],
                         lambda delta,out,o,y: -delta * np.log(_softMax(o)) / y.shape[0]],# TODO
    'mean':             [lambda delta,out,x : (delta + 0.0*x)/x.shape[0] ],# TODO
    }

# Unit tests for the functions. Run by `python functions.py`.
if __name__ == '__main__':
    x = np.array([
        [ 0.76677119,  0.12815245],
        [ 0.4007303 ,  0.77046941],
        [ 0.00574018,  0.71242641]])
    y = np.array([
        [-0.06655641,  0.10877971],
        [ 0.13663944, -0.12461873]])
    z = np.array([[0., 1.], [0., 1.], [1., 0.]])
    v =np.array([[ 0.96894013], [ 0.07382228]])
    # Eval mul
    expected_x_mul_y =  np.array([[-0.03352286,  0.06743895],
        [ 0.07860534, -0.05242359],
        [ 0.0969635 , -0.08815726]])
    np.testing.assert_allclose(EVAL_FUNS['mul'](x, y), expected_x_mul_y)
    expected_relu_y = np.array([
        [ 0.        ,  0.10877971],
        [ 0.13663944,  0.        ]])
    # Eval relu
    np.testing.assert_allclose(EVAL_FUNS['relu'](y), expected_relu_y)
    expected_softMax_x = np.array([
        [ 0.65444116,  0.34555884],
        [ 0.40860406,  0.59139594],
        [ 0.33033148,  0.66966852]])
    # Eval softMax
    np.testing.assert_allclose(EVAL_FUNS['softMax'](x), expected_softMax_x)
    expected_crossEnt_softMax_x_z = np.array([
        [ 1.06259235],
        [ 0.52526954],
        [ 1.10765864]])
    # Eval crossEnt
    pdb.set_trace()
    np.testing.assert_allclose(EVAL_FUNS['crossEnt'](expected_softMax_x, z), expected_crossEnt_softMax_x_z)
    # Eval mean
    expected_mean_v = 0.52138120499999996
    np.testing.assert_allclose(EVAL_FUNS['mean'](v), expected_mean_v)
    # BP mul
    delta_x_mul_y = np.array([
        [ 0.12523631,  0.00680066],
        [ 0.48109275,  0.95663136],
        [ 0.40436419,  0.56481742]])
    np.testing.assert_allclose(BP_FUNS['mul'][0](delta_x_mul_y, expected_x_mul_y, x, y), np.array([
        [-0.00759551,  0.01626473],
        [ 0.07204228, -0.05347794],
        [ 0.03452765, -0.01513473]]), rtol=1e-06)
    np.testing.assert_allclose(BP_FUNS['mul'][1](delta_x_mul_y, expected_x_mul_y, x, y), np.array([
        [ 0.29113716,  0.39180788],
        [ 0.67479632,  1.14031757]]))
    # BP relu
    delta_relu_y = np.array([
        [ 0.66202207,  0.59765468],
        [ 0.01812402,  0.58537534]])
    np.testing.assert_allclose(BP_FUNS['relu'][0](delta_relu_y, expected_relu_y, y), np.array([
        [ 0.        ,  0.59765468],
        [ 0.01812402,  0.        ]]))
    # BP crossEnt-softMax
    delta_crossEnt_softMax_x_z = np.array([
        [  5.69906247e-01],
        [  8.66851385e-01],
        [  2.79581480e-04]])
    np.testing.assert_allclose(BP_FUNS['crossEnt-softMax'][0](delta_crossEnt_softMax_x_z, expected_crossEnt_softMax_x_z, x, z), np.array([
        [  3.72970104e-01,  -3.72970104e-01],
        [  3.54198998e-01,  -3.54198998e-01],
        [ -1.87226917e-04,   1.87226917e-04]]))
    # BP mean
    pdb.set_trace()
    np.testing.assert_allclose(BP_FUNS['mean'][0](0.19950823, expected_mean_v, v), np.array([
        [ 0.09975412],
        [ 0.09975412]]))