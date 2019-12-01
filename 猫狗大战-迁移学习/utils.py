
# coding: utf-8

# In[1]:
import os
import tensorflow as tf
import numpy as np
from vgg_preprocess import preprocess_for_train
# vgg_preprocess.py里面对图像进行了预处理，源码先不研究# 数据输入

# 对图像数据加标签
# 猫狗数据集无标签，只是放在两个不同的文件夹下面
# 这里下载的数据集结构是 train/  test/ 训练集的文件名由cat/dog开头，测试集貌似没有标记
def get_file(file_dir):
    images = []
    labels=[]
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root, name))
            # 我们先令猫=0狗=1
            if 'cat' in name:
                labels.append(0)
            else:
                labels.append(1)
    
    # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()
    
    np.random.shuffle(temp)
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(float(i)) for i in label_list]
    
    return image_list, label_list


# In[2]:


img_width = 224
img_height = 224

# 通过读取列表来载入批量图片及标签
def get_batch(image_list, label_list, img_width, img_height, batch_size, capacity):
    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)
    
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    
    image = tf.image.decode_jpeg(image_contents, channels=3)
    #image, output_height, output_width, resize_side_min, resize_side_max
    image = preprocess_for_train(image,img_height, img_width )
    
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    
    return image_batch, label_batch


# In[3]:

# 标签转为独热编码
def onehot(labels):
    n_sample = len(labels)
    #n_class = max(labels) + 1
    n_class = 2 #猫狗只有两类
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    
    return onehot_labels
    

