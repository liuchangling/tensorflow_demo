# 介绍
2015年 google 公布的技术

CNN  根据预测结果和真实值之间的误差，调整卷积神经网络的CNN参数
Deep Dream 根据预测结果，和输入的期望结果之间的误差，调整输入图像的像素值。
注意：Deep Dream中 图像识别模型选择好之后，不做修改，仅修改输入图像。即**网络固定**


技术原理：
- 最大化输出层某一类别的概率，并输出图片。
- 最大化卷积层某一通道的激活值


# 经典神经网络 

## AlexNet
2012 ImageNet冠军
1. 防止过拟合 
数据增强
在数据集较小时获取更多的训练集
- 水平翻转
- 随机裁剪
- 修改对比度
dropout
2. GPU实现 网络分布在2个GPU上，且GPU之间某些层能相互通信
3. 非线性激活 ReLu代替了sigmoid 随机梯度下降速度大大加快
4. 大数据训练 120w数据集，1000个分类

## TensorFlow图像预处理函数
1. 解码
tf.image.decode_jpeg(img_data)
- tf.image.decode_png     
- tf.image.decode_gif    
- tf.image.decode_image
2. 缩放
tf.image.resize_images(img_data, [256, 256], method=0)
method:
- 0 双线性插值 BILINEAR
- 1 最近邻插值 NEAREST_NEIGHBOR
- 2 双立方插值 BICUBIC
- 3 像素区域插值 AREA
3. 截取或填充
tf.image.resize_image_with_crop_or_pad(image, target_height, target_width)
如果target小于原始图像，则在中心位置裁剪，否则利用黑色像素填充
4. 随机裁剪
tf.image_random_crop(image, size, seed=None, name=None)
5. 左右翻转
tf.image.flip_left_right(image_data)
6. 上下翻转
tf.image.flip_up_down(image_data)
7. 改变对比度
tf.image.random_contrast(image_data, lower=0.2, upper=3) 在0.2-3内随机对比度
tf.image.adjust_contrast(image_data, 0.5) 对比度*0.5
8. 白化处理
图像的像素值转换为0均值和单位方差
tf.image.per_image_standardization(image_data) 

### dropout
目的： 减少过拟合，缺点：增加训练时间
- 用于全连接层
- 每次迭代以某概率将神经元的输出置零，不参与前向和后向传播
- 产生不同的网络结构，进行组合，大大减少了过拟合
- 缺点：训练时间增加

## VGGNet
2014年的亚军。卷积核3*3，层数11-19。
卷积核大部分采用了逐层递增的方式。
现在常用VGG-16, VGG-19作为图像识别的预处理模型。在19层后VGG就达到了训练的瓶颈

## GoogleNet
2015冠军。
深度22层，宽度在每隔几个就定义了一个损失函数，避免梯度消失。
在卷积之前还用了1*1的卷积核进行降维

## ResNet 
2015年冠军 = VGG16 + GoolgeNet
深度152层。
层数增加时，常常导致梯度消失，导致无法向前调整。也称为网络退化。

### 捷径连接 shortcut connections
原理是残差映射学习较简单
判断F(x) = 0 比判断 H(x) = x 容易 其中 F(x) = H(x) - x.
所以输入层可以跳层直接和卷积结果相加。


# 导入模型
如何将训练好的经典模型导入？

### 生成检查点文件 
- .ckpt(checkpoint file)
- 通过tf.train.Saver()对象上调用
- Saver.save() 保存
- Saver.restore()加载

### 生成图协议文件
- .bp(graph proto file) 二进制文件
- tf.train.write_graph() 保存
- tf.import_graph_def() 加载


// TODO 一大堆代码

tf.expand_dims(data, index) 在index处插入一维


## 图的基本操作
- 建立 tf.Graph()
- 获得 tf.get_default_graph()
- 重置 tf.reset_default_graph()
- 获取所在图 	xxx.graph
- 获取张量  		g.get_tensor_by_name(name="xxx")


// TODO 一大堆代码进行Deep Dream 图像生成