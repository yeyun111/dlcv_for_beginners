## Caffe中进行模型融合
Blog: 
[在Caffe中实现模型融合](http://www.cnblogs.com/frombeijingwithlove/p/6683476.html)

## 从头训练两个不同模型
### step 1
> ./download_mnist.sh   

### step 2 
> python convert_mnist.py

### step 3
> python gen_img_list.py

### step 4

train with lenet_odd_solver.prototxt & lenet_even_solver.prototxt

## 基于预训练模型直接融合
预训练模型下载地址：  
https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/mnist_lenet_even_iter_30000.caffemodel  
https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/mnist_lenet_odd_iter_30000.caffemodel

## 生成融合后的prototxt

> python rename_n_freeze_layers.py input_model output_model prefix

拷贝从data层之后开始到最后要进行融合的特征层，比如ip1位置的prototxt，放到一个prototxt中，然后在开始加上data层，融合层可以用concat进行拼接，然后再接fc或是其他操作。

## 生成融合后的权重
> python fuse_model.py

## 基于融合的模型继续进行finetune

直接训练即可