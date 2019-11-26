# 介绍
CIFAR 有10个分类 每类有6k张32*32的彩色图片。
其中训练集5w个，测试集1w个。

与MNIST相比，CIFAR-10的特点：
1. 3通道rgb，而非灰度图
2. 尺寸32*32，大于28*28
3. 物体比例和特征不尽相同，噪声大，识别难度高

标记的结果是0-9，每个数字代表一个分类。
处理时现需要把数字转成独热编码。
比如  6 -> [0,0,0,0,0,0,1,0,0]

# 代码部分
1. 下载数据集
[下载地址](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
```python
import urllib.request
urllib.request.urlretrieve(url,filepath)
```
2. 解压数据
```python
import tarfile
tfile = tarfile.open('data/cifar-10-python.tar.gz', 'r:gz')
result = tfile.extractall('data/')
```
3. 解析样本，分别存入images，labels
```python
import pickle as p
with open(filename, 'rb') as f:
    # 一个样本由标签和图像(32*32*3 = 3072)组成
    data_dict = p.load(f, encoding = 'bytes')
	images = data_dict[b'data']
    labels = data_dict[b'labels']
```
4. 调整数据至bwhc结构，准备调用TensorFlow接口
这里我们得到的是Xtrain,Ytrain, Xtest, Ytest
5. 查看image&label的可视化接口
plt.imshow(Xtrain[6])
print(Ytrain[6])
6. 显示一组图像的可视化接口
```python
%matplotlib inline
import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(12,6)
....
遍历+imshow
...
plt.show()
```
7. 对X图像归一化
这时图像的rgb值还是0~255, 我们需要对其进行归一化处理，所以对每个值除以255

```python
# 图像进行数字标准化
Xtrain_normalize = Xtrain.astype('float32') / 255.0
Xtest_normalize = Xtest.astype('float32') / 255.0
```

8. Y标签onehot
CIFAR-10数据集的Y标签用0-9表示10分类，转成独热编码更方便我们操作

9. **【重点】定义网络结构**
定义了一系列卷积层，池化层，全连接层，详见代码。
全连接层用了 h_dropout = tf.nn.dropout(h, keep_prob = 0.8) dropout层
定义了pred = tf.nn.softmax(tf.matmul(h_dropout, W4) + b4)
10. 构建模型（同多层神经网络）
定义了损失函数，最小化pred和y的距离
定义了优化器optimizer，最小化损失函数，这次选的优化器是tf.train.AdamOptimizer
准确率函数
11. **【重点】实现断点续训**
- 生成 tf.train.Saver(max_to_keep=1)
- 读取 ckpt = tf.train.latest_checkpoint(ckpt_dir)
- 恢复 saver.restore(sess, ckpt)
- 保存 saver.save(sess, ckpt_dir + 'CIFAR10_cnn_model.cpkt', global_step = ep+1)

这个功能能保存或者恢复每次训练的结果，非常好用。可以随时增加训练的epoch轮数。
12. 训练（同多层神经网络）
即 sess.run(optimizer) 过程中统计loss&acc,以便后续可视化
13. 可视化展示loss&acc（同多层神经网络）
14. 模型评估与模型预测（同多层神经网络）