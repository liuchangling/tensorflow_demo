<script>
	var fetchWechat = require('fetch-wechat');
	var tf = require('@tensorflow/tfjs-core');
	var plugin = requirePlugin('tfjsPlugin');

	export default {
		onLaunch: function() {
			plugin.configPlugin({
				// polyfill fetch function
				fetchFunc: fetchWechat.fetchFunc(),
				// inject tfjs runtime
				tf,
				// provide webgl canvas
				canvas: wx.createOffscreenCanvas()
			});
		},
		onShow: function() {
			console.log('App Show')
		},
		onHide: function() {
			console.log('App Hide')
		}
	}
</script>

<style>
	/*每个页面公共css */
	html,
	body{
		height: 100%;
		width : 100%;
	}
</style>
