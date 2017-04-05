## Metric Learning with Siamese Network
### step 1
按照第八章准备MNIST数据的方法生成图片，然后执行
> ln -s /path/to/mnist mnist

在目录下链接mnist图片所在的目录

### step 2
> python gen_pairwise_imglist.py

生成成对图片的列表

### step 3
> /path/to/caffe/build/tools/convert_imageset ./ train.txt train_lmdb --gray  
> /path/to/caffe/build/tools/convert_imageset ./ train_p.txt train_p_lmdb --gray  
> /path/to/caffe/build/tools/convert_imageset ./ val.txt val_lmdb --gray  
> /path/to/caffe/build/tools/convert_imageset ./ val_p.txt val_p_lmdb --gray  

生成lmdb

### step 4
> /path/to/caffe/build/tools/caffe train -solver mnist_siamese_solver.prototxt -log_dir ./ 

训练模型

### step 5

> python visualize_result.py

进行结果可视化

## 预训练模型下载链接
https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/mnist_siamese_iter_20000.caffemodel
或  
http://pan.baidu.com/s/1qYk5MDQ