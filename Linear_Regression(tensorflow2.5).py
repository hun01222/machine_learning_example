import tensorflow as tf
import numpy as np
x_data=[[1,1],[2,2],[3,3]]
y_data=[[10],[20],[30]]
x_data=np.array(x_data, dtype=np.float32)
y_data=np.array(y_data, dtype=np.float32)
W=tf.Variable(tf.random.normal([2,1]))
b=tf.Variable(tf.random.normal([1]))

def run_optimization():
    with tf.GradientTape() as g:
        model=tf.matmul(x_data, W)+b
        cost=tf.reduce_mean(tf.square(model-y_data))
    gradients=g.gradient(cost, [W,b])
    tf.optimizers.SGD(0.01).apply_gradients(zip(gradients, [W,b]))

for step in range(2001):
    run_optimization()
    if step%500==0:
        model=tf.matmul(x_data, W)+b
        cost=tf.reduce_mean(tf.square(model-y_data))
        print("step: ", step, "W: ", W.numpy(), "b: ", b.numpy(), "Cost: ", cost.numpy())