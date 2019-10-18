1. 微信小程序
- 下载安装微信开发者工具的稳定版。
- 申请开发者账号等等。
- [微信tensorflow插件地址](https://mp.weixin.qq.com/wxopen/plugindevdoc?appid=wx6afed118d9e81df9)

2. [安装依赖](https://mp.weixin.qq.com/wxopen/plugindevdoc?appid=wx6afed118d9e81df9#tensorflow-js-)
```bash
cnpm install @tensorflow/tfjs-core  @tensorflow/tfjs-converter fetch-wechat --save
```
3. 添加插件到小程序
- 在小程序后台授权该插件
- 在微信开发者工具和开发填写对应的小程序id
- app.json中注册插件， 如果用uniapp ，在[manifest中配置](https://uniapp.dcloud.io/collocation/manifest?id=mp-weixin)
- 小程序开发工具-详情-本地设置-调试基础库 选择最新的版本，低版本不可用(有createObjectURL的问题)
- 小程序开发工具-设置-通用设置-打开gpu加速
 
4. 模型地址
- 谷歌云的base url是 https://storage.googleapis.com
- 中国镜像的base url是https://www.gstaticcnapps.cn

5. 加载模型咯
[tensorflow js模型](https://tensorflow.google.cn/js/models)
这里用的是posent 姿态检测模型
cnpm install @tensorflow-models/posenet --save

```javascript
const net = await posenet.load({
			  architecture: 'MobileNetV1',
			  outputStride: 16,
			  inputResolution: 193,
			  multiplier: 0.5
			});
```

如果google域名被拒绝，要在小程序后台开发设置的服务器域名中加入域名
现在google域名已备案，不需要了。

6. 让小程序支持async等异步操作
cnpm install regenerator-runtime --save
并在编译和小程序开发工具中去掉es6转es5的配置

7 