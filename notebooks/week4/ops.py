import tensorflow as tf


def deconv2d(x, output_shape, k_w=5, k_h=5, s_w=2, s_h=2, name=None):
    """Computes Deconvolution Operation
    
    Args:
        x: input tensor of shape (batch_size, width_in, height_in, channel_in)
        output_shape: list corresponding to [batch_size, width_out, height_out, channel_out]
        k_w: kernel width size; default is 5
        k_h: kernel height size; default is 5
        s_w: stride size for width; default is 2
        s_h: stride size for heigth; default is 2
        
    Returns:
        out: output tensor of shape (batch_size, width_out, hegith_out, channel_out)
    """
    
    channel_in = tf.shape(inputs)[-1]
    channel_out = tf.shape(output_shape)[-1]
    
    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=[k_w, k_h, channel_out, channel_in], 
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable('b', shape=[channel_out], initializer=tf.constant_initializer(0.0))
        
        out = tf.nn.conv2d_transpose(x, filter=w, output_shape=output_shape, strides=[1, s_w, s_h, 1]) + b
    
    return out