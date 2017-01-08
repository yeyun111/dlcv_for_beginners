## Prepare Data
### step 1
run convert_mnist.py to generate jpg images
### step 2
> python gen_caffe_imglist.py mnist/train train.txt

> python gen_caffe_imglist.py mnist/val val.txt

> python gen_caffe_imglist.py mnist/test test.txt

to get the image list for lmdb generation. Then run 
> /path/to/caffe/built/tools/convert_imageset ./ train.txt train_lmdb --gray --shuffle

> /path/to/caffe/built/tools/convert_imageset ./ val.txt val_lmdb --gray --shuffle

> /path/to/caffe/built/tools/convert_imageset ./ test.txt test_lmdb --gray --shuffle


to generate lmdb.

### step 3
> python gen_mxnet_imglist.py mnist/train train.lst

> python gen_mxnet_imglist.py mnist/val val.lst

> python gen_mxnet_imglist.py mnist/test test.lst

to generate image list for ImageRecordio. Then run 

> /path/to/mxnet/bin/im2rec train.lst ./ train.rec color=0

> /path/to/mxnet/bin/im2rec val.lst ./ val.rec color=0

> /path/to/mxnet/bin/im2rec test.lst ./ test.rec color=0

to generate ImageRecordio files.

## MXNet
run *train_lenet5.py* in mxnet directory to train model.

run *score_model.py* to score the model on test dataset.

run *benchmark_model.py* to benchmark model's performance.

run *recognize_digit.py* followed by path to handwritten digits images to recognized digits.

## Caffe
*lenet_train_val.prototxt* & *lenet_train_val_aug.prototxt* are the model definition for training original dataset and augmented dataset respectively.

*lenet_solver.prototxt* & *lenet_solver_aug.prototxt* are the solver for training original dataset and augmented dataset respectively.

*lenet_test.prototxt* is the model definition for testing on test dataset.

*lenet.prototxt* is the model definition for deployment and benchmark.

run *recognize_digit.py* followed by test image list to handwritten digits images to recognized digits.