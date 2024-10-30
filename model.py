import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Lambda, Concatenate
from utils import *
import math

def lrelu(x, trainbable=None):
    return tf.maximum(x * 0.2, x)

def relu(x, trainbable=None):
    return tf.maximum(x * 0,x)

def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name, trainable=True):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool_size = 2
        deconv_filter = tf.get_variable('weights', [pool_size, pool_size, output_channels, in_channels], trainable=True)
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1],
                                        name=scope_name)

        deconv_output = tf.concat([deconv, x2], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2])

        return deconv_output


def concat(layers):
    return tf.concat(layers, axis=3)

class L_exp(tf.keras.layers.Layer):

    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        self.pool = tf.keras.layers.AveragePooling2D(pool_size=(patch_size, patch_size))
        self.mean_val = mean_val

    def call(self, inputs, **kwargs):
        x = inputs
        b, h, w, c = x.get_shape().as_list()
        x = tf.reduce_mean(x, axis=3, keep_dims=True)
        mean = self.pool(x)

        d = tf.reduce_mean(tf.square(mean - tf.constant([self.mean_val], dtype=tf.float32), name='mean_val_square'), axis=[1, 2])
        return d

def l_spa(org, enhance):

    # Calculate the mean of input tensors
    org_mean = tf.reduce_mean(org, axis=3, keepdims=True)
    enhance_mean = tf.reduce_mean(enhance, axis=3, keepdims=True)

    # Depthwise average pooling
    org_pool = tf.nn.avg_pool(org_mean, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    enhance_pool = tf.nn.avg_pool(enhance_mean, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

    org_pool = tf.tile(org_pool, [1, 1, 1, 1])
    enhance_pool = tf.tile(enhance_pool, [1, 1, 1, 1])

    kernel_left = tf.constant([[[[0.0]], [[-1.0]], [[0.0]]],
                               [[[0.0]], [[1.0]], [[0.0]]],
                               [[[0.0]], [[0.0]], [[0.0]]]], dtype=tf.float32)

    kernel_right = tf.constant([[[[0.0]], [[0.0]], [[0.0]]],
                                [[[0.0]], [[1.0]], [[-1.0]]],
                                [[[0.0]], [[0.0]], [[0.0]]]], dtype=tf.float32)

    kernel_up = tf.constant([[[[0.0]], [[-1.0]], [[0.0]]],
                             [[[0.0]], [[1.0]], [[0.0]]],
                             [[[0.0]], [[0.0]], [[0.0]]]], dtype=tf.float32)

    kernel_down = tf.constant([[[[0.0]], [[0.0]], [[0.0]]],
                               [[[0.0]], [[1.0]], [[0.0]]],
                               [[[0.0]], [[-1.0]], [[0.0]]]], dtype=tf.float32)

    # 卷积操作

    # Convolution operations
    conv_org_left = tf.nn.conv2d(org_pool, kernel_left, strides=[1, 1, 1, 1], padding='SAME')
    conv_org_right = tf.nn.conv2d(org_pool, kernel_right, strides=[1, 1, 1, 1], padding='SAME')
    conv_org_up = tf.nn.conv2d(org_pool, kernel_up, strides=[1, 1, 1, 1], padding='SAME')
    conv_org_down = tf.nn.conv2d(org_pool, kernel_down, strides=[1, 1, 1, 1], padding='SAME')

    conv_enhance_left = tf.nn.conv2d(enhance_pool, kernel_left, strides=[1, 1, 1, 1], padding='SAME')
    conv_enhance_right = tf.nn.conv2d(enhance_pool, kernel_right, strides=[1, 1, 1, 1], padding='SAME')
    conv_enhance_up = tf.nn.conv2d(enhance_pool, kernel_up, strides=[1, 1, 1, 1], padding='SAME')
    conv_enhance_down = tf.nn.conv2d(enhance_pool, kernel_down, strides=[1, 1, 1, 1], padding='SAME')

    # Calculate E using the specified formula
    D_left = tf.square(conv_org_left - conv_enhance_left)
    D_right = tf.square(conv_org_right - conv_enhance_right)
    D_up = tf.square(conv_org_up - conv_enhance_up)
    D_down = tf.square(conv_org_down - conv_enhance_down)
    E = D_left + D_right + D_up + D_down

    #print(E)
    return E

#############################################################################
#############################################################################
# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     return Conv2D(out_planes, (3, 3), strides=stride, padding='same', groups=groups, use_bias=False, dilation_rate=dilation)
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    # Define a custom convolution layer that simulates the 'groups' parameter
    def custom_conv3x3(inputs):
        # Split the input tensor into 'groups' equal parts along the channel axis
        input_groups = tf.split(inputs, groups, axis=-1)
        output_groups = []

        # Apply 3x3 convolution separately to each group
        for group in input_groups:
            conv_group = tf.layers.conv2d(group,out_planes // groups, kernel_size=(3, 3), strides=stride, padding='same', use_bias=False, dilation_rate=dilation)
            output_groups.append(conv_group)

        # Concatenate the group-wise convolution results
        output = tf.concat(output_groups, axis=-1)
        return output

    return custom_conv3x3

def conv1x1(inputs, out_planes, stride=1):
    out = tf.layers.conv2d(inputs, out_planes, kernel_size=(1, 1), strides=stride, padding='same', use_bias=False)
    return out

def bottleneck_block(inputs, inplanes, planes, stride=1, downsample=None, group_width=1, dilation=1):
    width = planes
    x = conv1x1(inputs, width, stride)
    x = tf.nn.relu(x)
    x = conv3x3(width, width, stride, width // min(width, group_width), dilation)(x)
    x = tf.nn.relu(x)

    x = conv1x1(x,planes)
    if downsample == 1:
        identity = tf.layers.conv2d(x,planes, kernel_size=(1, 1), strides=stride,padding='same', use_bias=False)
    else:
        identity = inputs

    x = tf.add(x, identity)
    x = tf.nn.relu(x)
    return x


def stage_forward(inputs, inplanes, planes, group_width, blocks, stride=1, dilate=False, cheap_ratio=0.5):
    dilation = 1
    previous_dilation = dilation

    if dilate:
        dilation *= stride
        stride = 1
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = 1

    base = bottleneck_block(inputs, inplanes, planes, stride, downsample, group_width, previous_dilation)
    m_list = [base]

    group_width2 = int(group_width * 0.75)
    raw_planes = int(planes * (1 - cheap_ratio) / group_width2) * group_width2
    cheap_planes = planes - raw_planes

    x = base[:, :, :, :raw_planes]
    x = bottleneck_block(x, raw_planes, raw_planes, group_width=group_width2, dilation=dilation)
    m_list.append(x)
    m = Concatenate(axis=-1)(m_list)

    merge = tf.layers.average_pooling2d(m, (1, 1), strides=(1, 1))
    merge = tf.layers.conv2d(merge,cheap_planes, kernel_size=(1, 1), strides=stride, padding='same', use_bias=False)
    #merge = BatchNormalization()(merge)
    merge = tf.nn.relu(merge)
    merge = tf.layers.conv2d(merge,cheap_planes, kernel_size=(1, 1), strides=stride, padding='same', use_bias=False)
    #merge = BatchNormalization()(merge)

    c = base[:, :, :, raw_planes:]
    cheap = tf.layers.conv2d(c,cheap_planes, kernel_size=(1, 1), strides=stride, padding='same', use_bias=False)
    cheap_relu = tf.nn.relu(cheap+merge)

    out = tf.concat([x, cheap_relu], axis=-1)
    out = bottleneck_block(out, planes, planes, group_width=group_width, dilation=dilation)

    return out


def eca_block(inputs, b=1, gama=2):
    # 输入特征图的通道数
    in_channel = inputs.shape[-1]
    # 根据公式计算自适应卷积核大小
    kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
    # 如果卷积核大小是偶数，就使用它
    if kernel_size % 2:
        kernel_size = kernel_size
    # 如果卷积核大小是奇数就变成偶数
    else:
        kernel_size = kernel_size + 1
    # [h,w,c]==>[None,c] 全局平均池化
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    # [None,c]==>[c,1]
    x = tf.keras.layers.Reshape(target_shape=(in_channel, 1))(x)
    # [c,1]==>[c,1]
    x = tf.keras.layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)
    # sigmoid激活
    x = tf.nn.sigmoid(x)
    # [c,1]==>[1,1,c]
    x = tf.keras.layers.Reshape((1, 1, in_channel))(x)
    # 结果和输入相乘
    outputs = tf.keras.layers.multiply([inputs, x])

    return outputs


def DecomNet(input, training=True, group_width = 4):
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        conv0 = tf.keras.layers.SeparableConv2D(filters=16, kernel_size=(3, 3), strides=(1,1), padding='same', activation=lrelu)(input)

        conv1 = stage_forward(inputs=conv0, inplanes=16, planes=32, group_width=group_width, blocks=1, stride=1, dilate=False, cheap_ratio=0.5)
        conv2 = stage_forward(inputs=conv1, inplanes=32, planes=32, group_width=group_width, blocks=1, stride=1, dilate=False, cheap_ratio=0.5)
        conv3 = stage_forward(inputs=conv2, inplanes=32, planes=32, group_width=group_width, blocks=1, stride=1, dilate=False, cheap_ratio=0.5)
        conv4 = stage_forward(inputs=concat([conv3, conv2]), inplanes=32+32, planes=32, group_width=group_width, blocks=1, stride=1, dilate=False, cheap_ratio=0.5)
        conv5 = stage_forward(inputs=concat([conv4, conv1]), inplanes=32+32, planes=32, group_width=group_width, blocks=1, stride=1, dilate=False, cheap_ratio=0.5)

        R_conv6 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1,1), padding='same', activation=lrelu)(conv5)
        R_conv7 = slim.conv2d(R_conv6, 1, [1, 1], rate=1, activation_fn=None)
        R_out = tf.sigmoid(R_conv7)  #### Reflectance Structure

        C_conv6 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1,1), padding='same', activation=lrelu)(conv5)
        C_conv7 = slim.conv2d(C_conv6, 3, [1, 1], rate=1, activation_fn=None)
        C_out = tf.sigmoid(C_conv7)  #### Reflectance Colors

        l_conv2 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1,1), padding='same', activation=lrelu)(conv5)
        l_conv3 = tf.concat([l_conv2, conv5], 3)
        l_conv4 = slim.conv2d(l_conv3, 1, [1, 1], rate=1, activation_fn=None)
        L_out = tf.nn.softplus(l_conv4)  #### Illumination

    return R_out, C_out, L_out

def LC_adjust(input_L ,input_C, channel=16, kernel_size=3,group_width = 4):
    with tf.variable_scope('LCadjust'):
        input_L = tf.concat([input_L, input_L, input_L], axis=3)

        shared_conv_layer1 = tf.keras.layers.SeparableConv2D(filters=channel, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=lrelu,name='shared_conv_layer1')
        shared_conv_layer2 = tf.keras.layers.SeparableConv2D(filters=channel, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=lrelu,name='shared_conv_layer2')
        L_conv0 = shared_conv_layer1(input_L)
        L_conv1 = shared_conv_layer2(L_conv0)

        C_conv0 = shared_conv_layer1(input_C)
        C_conv1 = shared_conv_layer2(C_conv0)

        L_conv2 = concat([relu(C_conv1 - L_conv1),L_conv1])
        C_conv2 = concat([relu(L_conv1 - C_conv1), C_conv1])

        L_conv3 = eca_block(L_conv2)
        C_conv3 = eca_block(C_conv2)


        L_conv4 = stage_forward(inputs=L_conv3, inplanes=channel, planes=channel*2, group_width=group_width, blocks=1, stride=1, dilate=False, cheap_ratio=0.5)
        L_conv5 = tf.keras.layers.SeparableConv2D(filters=channel*2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=lrelu, name='L_conv5')(L_conv4)
        L_conv6 = tf.keras.layers.SeparableConv2D(filters=channel, kernel_size=(3, 3), strides=(1, 1),padding='same', activation=lrelu, name='L_conv6')(L_conv5)
        L_conv7 = slim.conv2d(L_conv6, 1, [1, 1], rate=1, activation_fn=None)
        L_out = tf.nn.softplus(L_conv7)


        C_conv4 = stage_forward(inputs=C_conv3, inplanes=channel, planes=channel*2, group_width=group_width, blocks=1, stride=1, dilate=False, cheap_ratio=0.5)
        C_conv5 = tf.keras.layers.SeparableConv2D(filters=channel*2, kernel_size=(3, 3), strides=(1, 1), padding='same',activation=lrelu, name='C_conv5')(C_conv4)
        C_conv6 = tf.keras.layers.SeparableConv2D(filters=channel, kernel_size=(3, 3), strides=(1, 1),padding='same', activation=lrelu, name='C_conv6')(C_conv5)
        C_conv7 = slim.conv2d(C_conv6, 3, [1, 1], rate=1, activation_fn=None)
        C_out = tf.sigmoid(C_conv7)
    return L_out,C_out

def R_adjust(input_low ,input_R, channel=16, kernel_size=3,group_width = 8):
    with tf.variable_scope('Radjust'): #, reuse=tf.AUTO_REUSE
        input = concat([input_low, input_R])
        conv0 = tf.keras.layers.SeparableConv2D(filters=channel*2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=lrelu,name='Rconv0')(input)
        conv1 = stage_forward(inputs=conv0, inplanes=channel*2, planes=channel*2, group_width=group_width, blocks=1,stride=1, dilate=False, cheap_ratio=0.5)
        conv2 = stage_forward(inputs=conv1, inplanes=channel*2, planes=channel*2, group_width=group_width, blocks=1,stride=1, dilate=False, cheap_ratio=0.5)
        output = slim.conv2d(conv2, 1, 1, padding='same', activation_fn=None)
        output = tf.sigmoid(output)
    return output