# coding: utf-8
from __future__ import print_function
import os
import time
import random
# from skimage import color
from PIL import Image
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from glob import glob
import argparse
#from tensorflow.keras.models import load_model

tf.keras.backend.set_floatx('float32')

def total_variation_loss(x, TVLoss_weight=1):
    h_tv = tf.reduce_sum(tf.image.total_variation(x))
    return TVLoss_weight * h_tv

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=10, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
parser.add_argument('--train_data_dir', dest='train_data_dir', default='./dataset/train',
                    help='directory for training inputs')

#training = tf.placeholder_with_default(False, shape=(), name='training')
exp_loss = L_exp(16, 0.6)

args = parser.parse_args()

batch_size = args.batch_size
patch_size = args.patch_size
sess = tf.Session()

input_1 = tf.placeholder(tf.float32, [None, None, None, 3], name='input_1')
input_2 = tf.placeholder(tf.float32, [None, None, None, 3], name='input_2')

# the input of illumination adjustment net
input_C_1 = tf.placeholder(tf.float32, [None, None, None, 3], name='input_C_1')
input_C_2 = tf.placeholder(tf.float32, [None, None, None, 3], name='input_C_2')
input_L_1 = tf.placeholder(tf.float32, [None, None, None, 1], name='input_L_1')
input_L_2 = tf.placeholder(tf.float32, [None, None, None, 1], name='input_L_2')
input_R_1 = tf.placeholder(tf.float32, [None, None, None, 1], name='input_R_1')
input_R_2 = tf.placeholder(tf.float32, [None, None, None, 1], name='input_R_2')

input_1_hsv = tf.image.rgb_to_hsv(input_1)
input_1_h = tf.expand_dims(input_1_hsv[:, :, :, 0], -1)
input_1_v = tf.expand_dims(input_1_hsv[:, :, :, 2], -1)

input_2_hsv = tf.image.rgb_to_hsv(input_2)
input_2_h = tf.expand_dims(input_2_hsv[:, :, :, 0], -1)
input_2_v = tf.expand_dims(input_2_hsv[:, :, :, 2], -1)

[R_decom_1, C_decom_1, S_decom_1] = DecomNet(input_1, training=False)
[R_decom_2, C_decom_2, S_decom_2] = DecomNet(input_2, training=False)

# the output of decomposition network

decom_output_C1 = C_decom_1
decom_output_C2 = C_decom_2
decom_output_L1 = S_decom_1
decom_output_L2 = S_decom_2
decom_output_R1 = R_decom_1
decom_output_R2 = R_decom_2


# the output of SFuseNet
L_out,C_out=LC_adjust(input_L_1,input_C_1)
R_out = R_adjust(input_1, input_R_1)


score_input_grad_1_x = tf.abs(gradient_no_norm(low_pass(input_1_v), "x"))
score_input_grad_1_y = tf.abs(gradient_no_norm(low_pass(input_1_v), "y"))
score_input_grad_1 = score_input_grad_1_x + score_input_grad_1_y

score_input_grad_2_x = tf.abs(gradient_no_norm(low_pass(input_2_v), "x"))
score_input_grad_2_y = tf.abs(gradient_no_norm(low_pass(input_2_v), "y"))
score_input_grad_2 = score_input_grad_2_x + score_input_grad_2_y

input_patch_score = tf.nn.softmax(tf.concat(
    [tf.expand_dims(tf.reduce_mean(score_input_grad_1, axis=[1, 2, 3]) / 0.05, axis=-1),
     tf.expand_dims(tf.reduce_mean(score_input_grad_2, axis=[1, 2, 3]) / 0.05, axis=-1)], axis=-1))
input_1_patch_score = input_patch_score[:, 0:1]
input_2_patch_score = input_patch_score[:, 1:2]

# define loss
# SSIM Fusion loss
#S_SSIM_loss = (1.0 - tf.reduce_mean(tf.image.ssim(R_out, input_R_2, max_val=1.0)))
S_SSIM_loss = (1.0 - tf.reduce_mean(tf.image.ssim(((R_out * C_out) * L_out), input_2, max_val=1.0)))


Radjust_loss_total = 1 * S_SSIM_loss

with tf.name_scope('scalar'):
    tf.summary.scalar('Sfus_loss_total', Radjust_loss_total)
    tf.summary.scalar('input_1_patch_score', tf.reduce_mean(input_1_patch_score))
    tf.summary.scalar('input_2_patch_score', tf.reduce_mean(input_2_patch_score))

with tf.name_scope('image'):
    tf.summary.image('input_1', tf.expand_dims(input_1[1, :, :, :], 0))
    tf.summary.image('input_2', tf.expand_dims(input_2[1, :, :, :], 0))

summary_op = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./log/' + '/SFus_train', sess.graph, flush_secs=60)
lr = tf.placeholder(tf.float32, name='learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')

var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
var_Radjust = [var for var in tf.trainable_variables() if 'Radjust' in var.name]
var_LCadjust = [var for var in tf.trainable_variables() if 'LCadjust' in var.name]

saver_Decom = tf.train.Saver(var_list=var_Decom)
saver_Radjust = tf.train.Saver(var_list=var_Radjust)
saver_LCadjust = tf.train.Saver(var_list=var_LCadjust)


train_op_Radjust = optimizer.minimize(Radjust_loss_total, var_list=var_Radjust)
sess.run(tf.global_variables_initializer())
print("[*] Initialize model successfully...")

### load data
train_1_data = []
train_2_data = []

train_1_data_names = glob(args.train_data_dir + '/low/*.png')
train_1_data_names.sort()

train_2_data_names = glob(args.train_data_dir + '/high/*.png')
train_2_data_names.sort()

assert len(train_1_data_names) == len(train_2_data_names)
print('[*] Number of training data: %d' % len(train_1_data_names))

for idx in range(len(train_1_data_names)):
    im_1 = load_images_no_norm(train_1_data_names[idx])
    train_1_data.append(im_1)
    im_2 = load_images_no_norm(train_2_data_names[idx])
    train_2_data.append(im_2)

pre_decom_checkpoint_dir = './checkpoint/decom_net_retrain/'#decom_net_retrain
ckpt_pre = tf.train.get_checkpoint_state(pre_decom_checkpoint_dir)
if ckpt_pre:
    print('loaded ' + ckpt_pre.model_checkpoint_path)
    saver_Decom.restore(sess, ckpt_pre.model_checkpoint_path)
else:
    print('No pre_decom_net checkpoint!')


LCadjust_checkpoint_dir = './checkpoint/LCadjust/'
ckpt_pre_LCadjust = tf.train.get_checkpoint_state(LCadjust_checkpoint_dir)
if ckpt_pre_LCadjust:
    print('loaded ' + ckpt_pre_LCadjust.model_checkpoint_path)
    saver_LCadjust.restore(sess, ckpt_pre_LCadjust.model_checkpoint_path)
else:
    print('No LCadjust checkpoint!')


decomposed_1_C_data = []
decomposed_2_C_data = []
decomposed_1_L_data = []
decomposed_2_L_data = []
decomposed_1_R_data = []
decomposed_2_R_data = []

input_1_img_data = []
input_2_img_data = []

for idx in range(len(train_1_data)):
    input_img1 = np.expand_dims(train_1_data[idx], axis=0)
    CC1 = sess.run([decom_output_C1], feed_dict={input_1: input_img1})
    decom_output_C1_component = np.squeeze(CC1)
    input_img1_component = np.squeeze(input_img1)
    decomposed_1_C_data.append(decom_output_C1_component)

    LL1 = sess.run([decom_output_L1], feed_dict={input_1: input_img1})
    decom_output_L1_component = np.squeeze(LL1)
    decomposed_1_L_data.append(decom_output_L1_component)

    RR1 = sess.run([decom_output_R1], feed_dict={input_1: input_img1})
    decom_output_R1_component = np.squeeze(RR1)
    decomposed_1_R_data.append(decom_output_R1_component)

    input_1_img_data.append(input_img1_component)

for idx in range(len(train_2_data)):
    input_img2 = np.expand_dims(train_2_data[idx], axis=0)
    CC2= sess.run([decom_output_C2], feed_dict={input_2: input_img2})
    decom_output_C2_component = np.squeeze(CC2)
    input_img2_component = np.squeeze(input_img2)
    decomposed_2_C_data.append(decom_output_C2_component)

    LL2 = sess.run([decom_output_L2], feed_dict={input_2: input_img2})
    decom_output_L2_component = np.squeeze(LL2)
    decomposed_2_L_data.append(decom_output_L2_component)

    RR2 = sess.run([decom_output_R2], feed_dict={input_2: input_img2})
    decom_output_R2_component = np.squeeze(RR2)
    decomposed_2_R_data.append(decom_output_R2_component)

    input_2_img_data.append(input_img2_component)

train_CFuse_1_C_data = decomposed_1_C_data
train_CFuse_2_C_data = decomposed_2_C_data
train_LFuse_1_L_data = decomposed_1_L_data
train_LFuse_2_L_data = decomposed_2_L_data
train_RFuse_1_R_data = decomposed_1_R_data
train_RFuse_2_R_data = decomposed_2_R_data
train_Decom_1_img_data = input_1_img_data
train_Decom_2_img_data = input_2_img_data

print('[*] Number of training data: %d' % len(train_CFuse_1_C_data))

learning_rate = 0.0001
epoch = 4000
train_phase = 'Radjust'
numBatch = len(train_CFuse_1_C_data) // int(batch_size)
train_op = train_op_Radjust
train_loss = Radjust_loss_total
saver = saver_Radjust

checkpoint_dir = './checkpoint/Radjust/'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print("No Radjust Net pre model!")

start_step = 0
start_epoch = 0
iter_num = 0
print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

start_time = time.time()
image_id = 0
counter = 0
for epoch in range(start_epoch, epoch):
    for batch_id in range(start_step, numBatch):
        batch_input_1_img = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        batch_input_2_img = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        batch_input_1_C = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        batch_input_2_C = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        batch_input_1_L = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_input_2_L = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_input_1_R = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_input_2_R = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")

        for patch_id in range(batch_size):
            C_1_data = train_CFuse_1_C_data[image_id]
            C_2_data = train_CFuse_2_C_data[image_id]

            S_1_data = train_LFuse_1_L_data[image_id]
            S_1_expand = np.expand_dims(S_1_data, axis=2)
            S_2_data = train_LFuse_2_L_data[image_id]
            S_2_expand = np.expand_dims(S_2_data, axis=2)

            R_1_data = train_RFuse_1_R_data[image_id]
            R_1_expand = np.expand_dims(R_1_data, axis = 2)
            R_2_data = train_RFuse_2_R_data[image_id]
            R_2_expand = np.expand_dims(R_2_data, axis = 2)

            img_1_data = train_Decom_1_img_data[image_id]
            img_2_data = train_Decom_2_img_data[image_id]

            h, w, _ = train_CFuse_1_C_data[image_id].shape
            x = random.randint(0, h - patch_size)
            y = random.randint(0, w - patch_size)
            S_1_data_crop = S_1_expand[x: x + patch_size, y: y + patch_size, :]
            S_2_data_crop = S_2_expand[x: x + patch_size, y: y + patch_size, :]
            R_1_data_crop = R_1_expand[x : x+patch_size, y : y+patch_size, :]
            R_2_data_crop = R_2_expand[x : x+patch_size, y : y+patch_size, :]
            C_1_data_crop = C_1_data[x : x+patch_size, y : y+patch_size, :]
            C_2_data_crop = C_2_data[x : x+patch_size, y : y+patch_size, :]
            img_1_data_crop = img_1_data[x: x + patch_size, y: y + patch_size, :]
            img_2_data_crop = img_2_data[x: x + patch_size, y: y + patch_size, :]

            rand_mode = np.random.randint(0, 7)
            batch_input_1_L[patch_id, :, :, :] = data_augmentation(S_1_data_crop, rand_mode)
            batch_input_2_L[patch_id, :, :, :] = data_augmentation(S_2_data_crop, rand_mode)
            batch_input_1_R[patch_id, :, :, :] = data_augmentation(R_1_data_crop , rand_mode)
            batch_input_2_R[patch_id, :, :, :] = data_augmentation(R_2_data_crop, rand_mode)
            batch_input_1_C[patch_id, :, :, :] = data_augmentation(C_1_data_crop , rand_mode)
            batch_input_2_C[patch_id, :, :, :] = data_augmentation(C_2_data_crop, rand_mode)
            batch_input_1_img[patch_id, :, :, :] = data_augmentation(img_1_data_crop, rand_mode)
            batch_input_2_img[patch_id, :, :, :] = data_augmentation(img_2_data_crop, rand_mode)

            image_id = (image_id + 1) % len(train_CFuse_1_C_data)
        counter += 1
        Cout, Cdecom2, _, loss, summary_str = sess.run([C_out, input_C_2, train_op, train_loss, summary_op],
                                        feed_dict={input_1: batch_input_1_img, input_2: batch_input_2_img,
                                                   input_C_1: batch_input_1_C, input_C_2: batch_input_2_C,
                                                   input_L_1: batch_input_1_L, input_L_2: batch_input_2_L,
                                                   input_R_1: batch_input_1_R, input_R_2: batch_input_2_R,
                                                   lr: learning_rate})

        print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
              % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
        train_writer.add_summary(summary_str, counter)
        iter_num += 1
    if (epoch + 1) % 1000== 0:
        saver_Radjust.save(sess, checkpoint_dir + 'model.ckpt', global_step=epoch + 1)

print("[*] Finish training for phase %s." % train_phase)



