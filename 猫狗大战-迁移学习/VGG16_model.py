
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf

# 定义VGGnet模型类
#tf.reset_default_graph()
# 修改VGG模型： 全连接层的神经元个数；trainable参数变动
# （1）预训练的VGG是在ImageNet数据集上进行训练的，对1000个类别进行判定
#     若希望利用已训练模型用于其他分类任务，需要修改最后的全连接层     
# （2）在进行Finetuning对模型重新训练时，对于部分不需要训练的层可以通过设置trainable=False来确保其在训练过程中不会被修改权值
class vgg16:
    def __init__(self, imgs):
        self.parameters = [] # 在类的初始化时加入全局列表，将所需共享的参数加载进来
        self.imgs = imgs
        self.convlayers()  # 初始化卷积层
        self.fc_layers()   # 初始化全连接层
        self.probs = tf.nn.softmax(self.fc8)# 输出每个属于各个类别的概率值

    def saver(self): # 无需调整
        return tf.train.Saver()

    def maxpool(self,name,input_data): #每次缩小1/2 无需调整
        out = tf.nn.max_pool(input_data,[1,2,2,1],[1,2,2,1],padding="SAME",name=name)
        return out

    def conv(self,name, input_data, out_channel,trainable=False): # trainable参数变动
        in_channel = input_data.get_shape()[-1]
        
        with tf.variable_scope(name):  #根据不同的卷积层标记不同的变量命名
            
            # 获取参数
            kernel = tf.get_variable("weights", [3, 3, in_channel, out_channel], dtype=tf.float32, trainable=False) # trainable参数变动
            biases = tf.get_variable("biases", [out_channel], dtype=tf.float32, trainable=False) # trainable参数变动
            conv_res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")  
            res = tf.nn.bias_add(conv_res, biases)
            out = tf.nn.relu(res, name=name)
        self.parameters += [kernel, biases] # 将卷积层定义的参数（kernel, biases）加入列表
        return out

    def fc(self,name,input_data,out_channel,trainable=True): # trainable参数变动
        shape = input_data.get_shape().as_list()
        if len(shape) == 4:
            size = shape[-1] * shape[-2] * shape[-3]  #展开成一维向量
        else:size = shape[1]
        input_data_flat = tf.reshape(input_data,[-1,size])
        with tf.variable_scope(name):
            weights = tf.get_variable(name="weights",shape=[size,out_channel],dtype=tf.float32, trainable=trainable) # trainable参数变动
            biases = tf.get_variable(name="biases",shape=[out_channel],dtype=tf.float32, trainable=trainable) # trainable参数变动
            res = tf.matmul(input_data_flat,weights)
            out = tf.nn.relu(tf.nn.bias_add(res,biases))
        self.parameters += [weights, biases] # 将全连接层定义的参数（weights, biases）加入列表
        return out

    def convlayers(self):
        #conv1 输入为self.imgs
        self.conv1_1 = self.conv("conv1re_1",self.imgs,64,trainable=False)# trainable参数变动
        self.conv1_2 = self.conv("conv1_2",self.conv1_1,64,trainable=False)# trainable参数变动
        self.pool1 = self.maxpool("poolre1",self.conv1_2)

        # 后续convk 的输入为convk-1
        #conv2 
        self.conv2_1 = self.conv("conv2_1",self.pool1,128,trainable=False)# trainable参数变动
        self.conv2_2 = self.conv("convwe2_2",self.conv2_1,128,trainable=False)# trainable参数变动
        self.pool2 = self.maxpool("pool2",self.conv2_2)

        #conv3
        self.conv3_1 = self.conv("conv3_1",self.pool2,256,trainable=False)# trainable参数变动
        self.conv3_2 = self.conv("convrwe3_2",self.conv3_1,256,trainable=False)# trainable参数变动
        self.conv3_3 = self.conv("convrew3_3",self.conv3_2,256,trainable=False)# trainable参数变动
        self.pool3 = self.maxpool("poolre3",self.conv3_3)

        #conv4
        self.conv4_1 = self.conv("conv4_1",self.pool3,512,trainable=False)# trainable参数变动
        self.conv4_2 = self.conv("convrwe4_2",self.conv4_1,512,trainable=False)# trainable参数变动
        self.conv4_3 = self.conv("conv4rwe_3",self.conv4_2,512,trainable=False)# trainable参数变动
        self.pool4 = self.maxpool("pool4",self.conv4_3)

        #conv5
        self.conv5_1 = self.conv("conv5_1",self.pool4,512,trainable=False)# trainable参数变动
        self.conv5_2 = self.conv("convrwe5_2",self.conv5_1,512,trainable=False)# trainable参数变动
        self.conv5_3 = self.conv("conv5_3",self.conv5_2,512,trainable=False)# trainable参数变动
        self.pool5 = self.maxpool("poorwel5",self.conv5_3)

    def fc_layers(self):
        # 全连接最后一层trainable = True
        self.fc6 = self.fc("fc1", self.pool5, 4096, trainable=False)# trainable参数变动
        self.fc7 = self.fc("fc2", self.fc6, 4096, trainable=False) # trainable参数变动
        #fc8正是我们需要训练的，因此trainable=True；2是n_class 即猫+狗二分类。fc8是我们重新定义的
        # 这就是所谓的finetuining 微调
        self.fc8 = self.fc("fc3", self.fc7, 2,trainable=True) 

    # 这个函数将获取的权重载入VGG模型中
    def load_weights(self, weight_file, sess):
        # 载入文件
        weights = np.load(weight_file) 
        # 对权值排序，准备删除最后一层
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i not in [30, 31]:# 剔除不需载入的层 即剔除fc8
                sess.run(self.parameters[i].assign(weights[k]))
        print("-----------weights loaded---------------")


# ## 载入权重
# 下载键值对文件，并剔除不需要的层，然后进行训练调整
# 
# [权重文件](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz )(这个有500+M)
# 
# [分类文件](https://www.cs.toronto.edu/~frossard/vgg16/imagenet_classes.py)

# In[ ]:



