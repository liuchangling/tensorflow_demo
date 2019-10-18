<template>
	<view class="content">
		<view class="status">{{status}}</view>
		<camera device-position="front" flash="off" binderror="error" class="camera">
			<canvas canvas-id="pose" class=canvas>

			</canvas>
		</camera>
	</view>
</template>

<script>
	const tf = require('@tensorflow/tfjs-core')
	const posenet = require('@tensorflow-models/posenet')
	const regeneratorRuntime = require('regenerator-runtime')
	export default {
		data() {
			return {
				scale: 0.6,
				status: '初始化成功...',
			}
		},
		async onReady() {
			// 获取camera
			const camera = uni.createCameraContext(this)
			this.canvas = uni.createCanvasContext("pose", this)

			this.loadPosenet();

			// 每5帧监听一次
			let count = 0;
			const listener = camera.onCameraFrame(frame => {
				count++
				if (count == 10) {
					count = 0
					if (this.net) {
						this.drawPose(frame)
					}
				}
			})

			listener.start()
		},
		methods: {
			async detectPose(frame, net) {
				const imgData = {
					data: new Uint8Array(frame.data),
					width: frame.width,
					height: frame.height
				}
				// 手机内存有限，边运行边清理中间环节变量
				const imgSlice = tf.tidy(() => {
					// 将4通道（RGBA)的图像转为tf张量处理
					const imgTensor = tf.browser.fromPixels(imgData, 4)
					//去掉alpha透明度的通道 只要rgb
					return imgTensor.slice([0, 0, 0], [-1, -1, 3])
				})

				// pose是一个Promise 内部有解析的结果 
				const pose = net.estimateSinglePose(imgSlice, {
					flipHorizontal: false
				})

				// 丢掉张量，释放内存
				imgSlice.dispose()

				return pose
			},
			async loadPosenet() {
				//  加载模型 也可以修改modelUrl把模型放到本地
				this.status = '开始从谷歌下载posenet...'
				this.net = await posenet.load({
					architecture: 'MobileNetV1',
					outputStride: 16,
					inputResolution: 193,
					multiplier: 0.5,
					modelUrl: 'https://www.gstaticcnapps.cn/tfjs-models/savedmodel/posenet/mobilenet/float/050/model-stride16.json',
				});
				console.log("Posenet load complete!!!")
				this.status = 'posenet加载成功! 开始体验吧!'
			},

			async drawPose(frame) {
				if (frame == null || this.canvas == null) return
				const pose = await this.detectPose(frame, this.net);
				if (pose.score > 0.3) {
					pose.keypoints.forEach(point => {
						if (point.score > 0.5) {
							const {
								y,
								x
							} = point.position;
							this.drawCircle(x, y)
						}
					})
				}
				const akPoints = posenet.getAdjacentKeyPoints(pose.keypoints, 0.5)
				akPoints.forEach(points => {
					this.drawLine(points[0], points[1])
				})
				this.canvas.draw()
			},

			drawCircle(x, y) {
				this.canvas.beginPath()
				this.canvas.arc(x * this.scale, y * this.scale, 3, 0, 2 * Math.PI)
				this.canvas.fillStyle = 'aqua'
				this.canvas.fill()
			},

			drawLine(p0, p1) {
				this.canvas.beginPath()
				this.canvas.moveTo(p0.x * this.scale, p0.y * this.scale)
				this.canvas.moveTo(p1.x * this.scale, p1.y * this.scale)
				this.canvas.lineWidth = 2
				this.canvas.strokeStyle = 'aqua'
				this.canvas.stroke()
			}
		}
	}
</script>

<style>
	.content {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		width: 100%;
		height: 100%;
	}

	.camera {
		width: 100%;
		height: 100%;
	}

	.canvas {
		width: 100%;
		height: 100%;
	}

	.status {
		width: 100%;
		line-height: 12px;
		font-size: 12px;
		height: 12px;
		text-align: center;
	}
</style>
