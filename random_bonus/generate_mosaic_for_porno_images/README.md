## 给XX图片生成马赛克
Blog: 
[提高驾驶技术：用GAN去除(爱情)动作片中的马赛克和衣服](https://zhuanlan.zhihu.com/p/27199954)

## step 1
下载yahoo的open_nsfw(Not Safe For Work)模型
> ./clone_open_nsfw.sh  

**注**：原版open_nsfw中执行global pooling的是最后一层kernel size为7的pooling层，为了实现输入大小可变，让马赛克更精细一些，deploy_global_pooling.prototxt中将
> kernel_size: 7   

改为了
> global_pooling: true

## step 2

> python gen_mosaic.py [input dir] [output dir]  

代码基于第10章的激活响应可视化：[visualize_activation.py](https://github.com/frombeijingwithlove/dlcv_for_beginners/blob/master/chap10/visualize_activation.py) 修改而来。

## step 3 (optional)
用于pix2pix训练的图片如果是长宽比较大，并且XX区域大都位于画面中心，可以考虑中央裁剪并重新缩放到256x256：
> python crop_n_resize [dir_0] [dir_1] ... [dir_n] 256
