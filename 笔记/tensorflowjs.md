1. 微信小程序
- 下载安装微信开发者工具的稳定版。
- 申请开发者账号等等。
- [微信tensorflow插件地址](https://mp.weixin.qq.com/wxopen/plugindevdoc?appid=wx6afed118d9e81df9)

2. [安装依赖](https://mp.weixin.qq.com/wxopen/plugindevdoc?appid=wx6afed118d9e81df9#tensorflow-js-)
- cnpm install @tensorflow/tfjs-core
- cnpm install @tensorflow/tfjs-converter
- cnpm install fetch-wechat

3. 模型地址
- 谷歌云的base url是 https://storage.googleapis.com
- 中国镜像的base url是https://www.gstaticcnapps.cn

4. posenet模型
urlpath /tfjs-models/savedmodel/posenet/mobilenet/float/050/model-stride16.json
[谷歌云地址](https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/float/050/model-stride16.json)
[中国镜像的地址](https://www.gstaticcnapps.cn/tfjs-models/savedmodel/posenet/mobilenet/float/050/model-stride16.json)
或者 cnpm install @tensorflow-models/posenet
