<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## 问题描述
已有标记好的7w数据集。每个数据包含一个28*28的手写数字图片以及标记好的结果（0-9）
入门将使用一个神经元完成训练。
[MNIST数据集下载地址](http://yann.lecun.com/exdb/mnist/)

TensorFlow提供的下载方法(下载太慢的话用上面的链接试试)
```python
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

#### 如何打印一个图片
```python
#按照28*28展示一个图片
def plot_image(image):
    plt.imshow(image.reshape(28,28), cmap='binary')
    plt.show()
plot_image(mnist.train.images[1])
```

数据集分为三类
train，validation，test
每类有image和label

## one hot encoding 独热编码
就是答案为1其他都是0的稀疏向量
常用于表示拥有有限个可能值的字符串或者标识符
#### 为什么要用独热编码
1. 将离散特征的取值扩展到了欧式空间，离散特征的某个取值就对应欧式空间的某个点
2. 机器学习算法中，特征之间的距离或相似度的常用计算方法都是基于欧式空间的
3. 将离散型特征使用独热编码，会让特征之间的距离计算更加合理

## 批量读取数据

```
batch_image_xs, batch_labels_ys = mnist.train.next_batch(batch_size=10)
```

next_batch内部会自动对数据集shuffle

## 了解random_normal
```
norm = tf.random_normal([100])

with tf.Session() as sess:
    norm_data = norm.eval()
    print(norm_data[:])
plt.hist(norm_data)
plt.show()
```

## Softmax分类
```
forward = tf.matmul(x,w) + b
pred = tf.nn.softmax(forward)
```

所谓的线性回归，就是处理连续空间的值。

所谓的逻辑回归，就是处理离散的值。分类问题是其中一种。

为什么我们要将forward进行softmax分类呢？

因为，我们需要将预测输出值控制在[0,1]区间内。

而且保证所有的概率之和为1。

公式如下
$$ p_i = {e^{y_i} \over {\sum^C_{k=1} e^{y_k}}}$$

我的疑问是，不用e求指数貌似也能完成这个需求啊？

## sigmod函数
又称S函数，常用的逻辑函数之一。

$$ y = {{1}\over{1 + e^{-z}}} $$

好处1 计算的结果在[0,1]内,可以作为一个概率，或者可信度。
好处2 这个函数求导方便
好处3 曲线两边平滑，中间陡峭，梯度大。


缺点，带入平方损失函数是一个非凸函数，有多个极小值。
如果采用梯度下降法，会容易导致陷入局部最优解之中

## 对数损失函数
为了解决平方损失函数的问题，我们用对数损失函数带入sigmod建立模型。

$$ J(w,b) = \sum_{(x,y)\in D} -ylog(y') - (1-y)log(1-y') $$

- x,y是样本集中的数据集。
- y是有标签样本中的标签，取值非0即1
- y’是预测值,在[0,1]之间

## 交叉熵
信息论中，表明两个概率曲线的距离或者相似度等概念。也可以作为损失函数。

公式如下
$$ H(p,q) = - \sum_x p(x) logq(x) $$

举个例子，假设一个三分类的答案是(1,0,0)
A模型预测结果(0.5,0.2,0.3)
B模型预测结果(0.7,0.1,0.2)
为0的部分相乘都是0
H((1,0,0),(0.5,0.2,0.3)) = -log 0.5 = 0.301
H((1,0,0),(0.7,0.1,0.2)) = -log 0.7 = 0.155

所以B比A优秀

交叉熵损失函数定义如下
$$ Loss = - \sum^n_{i=1} y_i log y_i' $$
其中带'为预测值，不带'为标签值。

代码如下
```
loss_function = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
```

## 准确率
分类问题中预测值和标签值是否相等

```
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

```

问题 tf.equal 和== 有啥区别？
tf.cast是类型转换，第二行那一段为投射为浮点数的意思

## argmax 理解
按轴压缩之后取出最大值的下标


## 可视化看结果
```
# 可视化看结果的函数，最多显示25张图
def plot_images_labels_prediction(images,
                                  labels,
                                  prediction,
                                  index, # 从第index个开始显示
                                  num=10):
    fig = plt.gcf()# get current figure
    fig.set_size_inches(10,12) # 当前图像大小为10英寸*12英寸
    if num>25:
        num = 25
    for i in range(0,num):
        ax = plt.subplot(5,5,i+1) # 获取当前要处理的子图
        ax.imshow(np.reshape(images[index],(28,28)),cmap='binary') #指定image reshape
        title = 'label=' + str(np.argmax(labels[index]))
        
        ax.set_title(title,fontsize=10) # 在图上显示预测值
        ax.set_xticks([]) # 为了美观， 不显示x和y轴
        ax.set_xticks([])
        index = index + 1
    plt.show()
    
```