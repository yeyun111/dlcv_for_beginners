import mxnet as mx

test_dataiter = mx.io.ImageRecordIter(
    path_imgrec="../data/test.rec",
    data_shape=(1, 28, 28),
    batch_size=100,
    mean_r=128,
    scale=0.00390625,
)

mod = mx.mod.Module.load('mnist_lenet', 35, context=mx.gpu(2))
mod.bind(
    data_shapes=test_dataiter.provide_data, 
    label_shapes=test_dataiter.provide_label, 
    for_training=False)

'''
# in case we need to continue to train from epoch 35
mod.fit(...,
        arg_params=arg_params,
        aux_params=aux_params,
        begin_epoch=35)
'''

metric = mx.metric.create('acc')
mod.score(test_dataiter, metric)

for name, val in metric.get_name_value():
    print('{}={:.2f}%'.format(name, val*100))
