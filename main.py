# imports
import numpy as np
import trax.layers as tl
from trax import shapes, fastmath

# defining ReLU layer
relu = tl.Relu()
print(relu.name, relu.n_in, relu.n_out)

x = np.array([-2,-1,0,1,2])
print(x)

y = relu(x)
print(f"Output of ReLU layer: {y}")

# defining concatenate layer
concat = tl.Concatenate(n_items=3)
x1 = np.array([1,2,3])
x2 = np.array([1,2,3])
x3 = np.array([1,2,3])
y = concat([x1,x2,x3])
print("Output of concat layer: {}".format(y))

# layers containing weights
normLayer = tl.LayerNorm()
x = np.arange(10).reshape(2,5)
normLayer.init(shapes.signature(x)) # shape of input need to be known earlier
print(f'Weights initialized: {normLayer.weights[0]}')
print(f'Bias initialized: {normLayer.weights[1]}')
print(f'Input shape: {shapes.signature(x)}')
print(f'Output after normalization: {normLayer(x)}')

# defining a custom layer
def TimesTwo(name):
    return tl.Fn(name, lambda x: x*2)

timesTwo = TimesTwo('double')
x = np.arange(10)
out = timesTwo(x)
print(f"1-10 doubled: {out}")
print(timesTwo.name, timesTwo.n_in, timesTwo.n_out)

# defining Serial element
serial = tl.Serial(
    tl.Relu(),
    tl.LayerNorm(),
    timesTwo,
    tl.Dense(2),
    tl.Dense(1),
    tl.LogSoftmax()
)

for i,layer in enumerate(serial.sublayers):
    print(f"{i+1} - {layer}")

x_in = fastmath.numpy.arange(10)
serial.init(shapes.signature(x_in))
y = serial(x_in)

print(f"Output is: {y}, shape is: {y.shape}")