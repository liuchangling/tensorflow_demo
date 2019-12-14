[api](https://js.tensorflow.org/api/latest/)
[tutorials](https://www.tensorflow.org/js/tutorials)

#### 张量
- 0阶 tf.scalar 
- 1阶 tf.tensor1d
- 2阶 tf.tensor2d

#### 变量
- 初始化 tf.variable
- 赋值 x.assign

# 构建模型
#### 手工
```javascript
function predict(input){
	// y = a * x + b
	return tf.tidy(()=>{
		const x = tf.scalar(input)
		const y = a.mul(x).add(b)
		
		return y
	})
}

const a = tf.scalar(2)
const b = tf.scalar(5)

const result = predict(2)
result.print()

```

#### tf.model
```javascript
// Define input, which has a size of 5 (not including batch dimension).
const input = tf.input({shape: [5]});

// First dense layer uses relu activation.
const denseLayer1 = tf.layers.dense({units: 10, activation: 'relu'});
// Second dense layer uses softmax activation.
const denseLayer2 = tf.layers.dense({units: 4, activation: 'softmax'});

// Obtain the output symbolic tensor by applying the layers on the input.
const output = denseLayer2.apply(denseLayer1.apply(input));

// Create the model based on the inputs.
const model = tf.model({inputs: input, outputs: output});

// The model can be used for training, evaluation and prediction.
// For example, the following line runs prediction with the model on
// some fake data.
model.predict(tf.ones([2, 5])).print();
```

#### 内存管理
- 张量.dispose() 内存释放
- tf.tidy() 设置一个作用域，里面函数执行完后，会释放内部资源，但会保存返回值