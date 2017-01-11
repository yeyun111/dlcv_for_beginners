## Prepare Data
运行 *gen_data.py* 产生随机数据并用pickle导出为文件

## MXNet
在mxnet文件夹下运行 *simple_mlp.py* 训练模型并进行结果可视化

## Caffe
### step 1
在caffe文件夹下运行 *gen_hdf5.py* 将数据转换为HDF5格式
### step 2
运行 *simple_mlp_train.py* 训练模型
### step 3
运行 *simple_mlp_test.py* 测试模型及可视化