import mxnet as mx
import logging

# data & preprocessing
data = mx.symbol.Variable('data')

# 1st conv
conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=20)
pool1 = mx.symbol.Pooling(data=conv1, pool_type="max",
                          kernel=(2, 2), stride=(2, 2))
# 2nd conv
conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), num_filter=50)
pool2 = mx.symbol.Pooling(data=conv2, pool_type="max",
                          kernel=(2, 2), stride=(2, 2))
# 1st fc & relu
flatten = mx.symbol.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
relu1 = mx.symbol.Activation(data=fc1, act_type="relu")

# 2nd fc
fc2 = mx.symbol.FullyConnected(data=relu1, num_hidden=10)
# loss
lenet5 = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

train_dataiter = mx.io.ImageRecordIter(
    path_imgrec="../data/train.rec",
    data_shape=(1, 28, 28),
    batch_size=50,
    mean_r=128,
    scale=0.00390625,
    rand_crop=True,
    min_crop_size=26,
    max_crop_size=28,
    max_rotate_angle=15,
    fill_value=0
)

val_dataiter = mx.io.ImageRecordIter(
    path_imgrec="../data/val.rec",
    data_shape=(1, 28, 28),
    batch_size=100,
    mean_r=128,
    scale=0.00390625,
)

logging.getLogger().setLevel(logging.DEBUG)
fh = logging.FileHandler('train_mnist_lenet.log')
logging.getLogger().addHandler(fh)

lr_scheduler = mx.lr_scheduler.FactorScheduler(1000, factor=0.95)
optimizer_params = {
    'learning_rate': 0.01,
    'momentum': 0.9,
    'wd': 0.0005,
    'lr_scheduler': lr_scheduler
}
checkpoint = mx.callback.do_checkpoint('mnist_lenet', period=5)

mod = mx.mod.Module(lenet5, context=mx.gpu(2))
mod.fit(train_dataiter,
        eval_data=val_dataiter,
        optimizer_params=optimizer_params,
        num_epoch=36,
        epoch_end_callback=checkpoint)
