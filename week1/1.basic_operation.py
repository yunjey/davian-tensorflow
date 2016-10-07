import tensorflow as tf

# open session 
sess = tf.Session()

print tf.add(1, 2)
# Tensor("Add:0", shape=(), dtype=int32)

print sess.run(tf.add(1, 2))
# 3

print tf.sub(3, 4)
# Tensor("Sub:0", shape=(), dtype=int32)

print sess.run(tf.sub(3, 4))
# -1


# create a constant 2 X 2 matrix
tensor_1 = tf.constant([[1., 2.], [3., 4.]])

tensor_2 = tf.constant([[5., 6.],[7., 8.]])

# create a matrix multiplication operation
output_tensor = tf.matmul(tensor_1, tensor_2)

result = sess.run(output_tensor)
print(result)
# [[19., 22.] 
#  [43., 50.]]

sess.close()