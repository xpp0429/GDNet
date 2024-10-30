# coding: utf-8
from __future__ import print_function
import os
import time
import random
from PIL import Image
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from glob import glob
from skimage import color, filters
import argparse
import scipy.io as scio

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.keras.backend.set_floatx('float32')  # 设置默认的浮点数精度为 float64


parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_SFus_dir', dest='save_SFus_dir', default='./Results/S_result/', help='directory for testing outputs')
parser.add_argument('--save_IFus_dir', dest='save_IFus_dir', default='./Results/F/',
                    help='directory for testing outputs')
parser.add_argument('--save_CFus_dir', dest='save_CFus_dir', default='./Results/C/',
                    help='directory for testing outputs')
parser.add_argument('--save_RFus_dir', dest='save_RFus_dir', default='./Results/R/',
                    help='directory for testing outputs')
parser.add_argument('--test_1_dir', dest='test_1_dir', default='./dataset/test/demo/low/',
                    help='directory for low inputs')
parser.add_argument('--test_2_dir', dest='test_2_dir', default='./dataset/test/demo/high/',
                    help='directory for high inputs')

args = parser.parse_args()

sess = tf.Session()
training = tf.placeholder_with_default(False, shape=(), name='training')
input_1_image = tf.placeholder(tf.float32, [None, None, None, 3], name='input_1_image')

[R_1, C_1, S_1] = DecomNet(input_1_image,training)

Fus_S,Fus_C = LC_adjust(S_1,C_1)
Fus_R =R_adjust(input_1_image,R_1)#R_1

Fus_image = Fus_R * Fus_S * Fus_C#Fus_R * Fus_C * Fus_S

# load pretrained model
var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
var_LCadjust = [var for var in tf.trainable_variables() if 'LCadjust' in var.name]
var_Radjust = [var for var in tf.trainable_variables() if 'Radjust' in var.name]

#print(var_LCadjust)
g_list = tf.global_variables()

saver_Decom = tf.train.Saver(var_list=var_Decom)
saver_LCadjust = tf.train.Saver(var_list=var_LCadjust)
saver_Radjust = tf.train.Saver(var_list=var_Radjust)

#saver_SFus = tf.train.Saver(var_list=tf.trainable_variables())

Decom_checkpoint_dir = './checkpoint/decom_net_retrain/'#decom_net_retrain
ckpt_pre = tf.train.get_checkpoint_state(Decom_checkpoint_dir)
if ckpt_pre:
    print('loaded ' + ckpt_pre.model_checkpoint_path)
    saver_Decom.restore(sess, ckpt_pre.model_checkpoint_path)
else:
    print('No IID checkpoint!')

LCadjust_checkpoint_dir = './checkpoint/LCadjust/'
ckpt_pre_LCadjust = tf.train.get_checkpoint_state(LCadjust_checkpoint_dir)
if ckpt_pre_LCadjust:
    print('loaded ' + ckpt_pre_LCadjust.model_checkpoint_path)
    saver_LCadjust.restore(sess, ckpt_pre_LCadjust.model_checkpoint_path)
else:
    print('No LCadjust checkpoint!')

Radjust_checkpoint_dir = './checkpoint/Radjust/'
ckpt_pre_Radjust = tf.train.get_checkpoint_state(Radjust_checkpoint_dir)
if ckpt_pre_Radjust:
    print('loaded ' + ckpt_pre_Radjust.model_checkpoint_path)
    saver_Radjust.restore(sess, ckpt_pre_Radjust.model_checkpoint_path)
else:
    print('No Radjust checkpoint!')

save_SFus_dir = args.save_SFus_dir
if not os.path.isdir(save_SFus_dir):
    os.makedirs(save_SFus_dir)

save_IFus_dir = args.save_IFus_dir
if not os.path.isdir(save_IFus_dir):
    os.makedirs(save_IFus_dir)

save_CFus_dir = args.save_CFus_dir
if not os.path.isdir(save_CFus_dir):
    os.makedirs(save_CFus_dir)

save_RFus_dir = args.save_RFus_dir
if not os.path.isdir(save_RFus_dir):
    os.makedirs(save_RFus_dir)


###load eval data
eval_1_data = []
eval_1_img_name = []

eval_1_data_name = glob(args.test_1_dir + '*')
eval_1_data_name.sort()



for idx in range(len(eval_1_data_name)):
    [_, name_1] = os.path.split(eval_1_data_name[idx])

    suffix_1 = name_1[name_1.find('.') + 1:]
    name_1 = name_1[:name_1.find('.')]

    eval_1_img_name.append(name_1)
    eval_1_im = load_images_no_norm(eval_1_data_name[idx])

    h,w,c = eval_1_im.shape
    h_tmp = h%1
    w_tmp = w%1
    eval_1_im = eval_1_im[0:h-h_tmp, 0:w-w_tmp, :]

    eval_1_data.append(eval_1_im)

Time_data = np.zeros(60)
print("Start evalating!")

for idx in range(len(eval_1_data)):
    print(idx)
    name_1 = eval_1_img_name[idx]

    input_1 = eval_1_data[idx]

    input_1_eval = np.expand_dims(input_1, axis=0)

    h, w, _ = input_1.shape
    time_start = time.time()
    C1, R_Fusion, S_Fusion, C_Fusion, Img_Fusion = sess.run([C_1, Fus_R, Fus_S, Fus_C, Fus_image ],
                                                        feed_dict={input_1_image: input_1_eval, training: False})

    #print(C_Fusion)
    time_end = time.time()
    Time_data[idx] = time_end - time_start
    save_images_S(os.path.join(save_SFus_dir, '%s.png' % (name_1)), S_Fusion)
    save_images_F(os.path.join(save_CFus_dir, '%s.png' % (name_1)), C_Fusion)
    save_images(os.path.join(save_IFus_dir, '%s.png' % (name_1)), Img_Fusion)
    save_images(os.path.join(save_RFus_dir, '%s.png' % (name_1)), R_Fusion)

scio.savemat('./Results/Fusion/time.mat', {'I': Time_data})




