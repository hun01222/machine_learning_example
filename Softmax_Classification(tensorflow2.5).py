import tensorflow as tf
import numpy as np
x_data=np.array([[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]], dtype=np.float32)
y_data=np.array([[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]], dtype=np.float32)
W=tf.Variable(tf.random.normal([4,3]))
b=tf.Variable(tf.random.normal([3]))

def run_optimization():
  with tf.GradientTape() as g:
    model_LC=tf.matmul(x_data, W)+b
    cost=tf.reduce_sum(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_LC, labels=y_data)))
  gradients=g.gradient(cost, [W,b])
  tf.optimizers.SGD(0.1).apply_gradients(zip(gradients, [W,b]))

for step in range(2001):
  run_optimization()
  if step%500==0:
    model_LC=tf.matmul(x_data, W)+b
    model=tf.argmax(tf.nn.softmax(model_LC), 1)
    cost=tf.reduce_sum(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_LC, labels=y_data))).numpy()
    accuracy=tf.reduce_mean(tf.cast(tf.equal(model, tf.argmax(y_data, 1)), tf.float32))
    print("Epoch {:4d} Accuracy: {:.3f}, Cost: {:.3f}".format(step, accuracy, cost))