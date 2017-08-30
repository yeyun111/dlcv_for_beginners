import sys
sys.path.append('/path/to/caffe/python')
import caffe

fusion_net = caffe.Net('lenet_fusion_train_val.prototxt', caffe.TEST)

model_list = [
    ('even', 'lenet_even_train_val.prototxt', 'mnist_lenet_even_iter_30000.caffemodel'),
    ('odd', 'lenet_odd_train_val.prototxt', 'mnist_lenet_odd_iter_30000.caffemodel')
]

for prefix, model_def, model_weight in model_list:
    net = caffe.Net(model_def, model_weight, caffe.TEST)

    for layer_name, param in net.params.iteritems():
        n_params = len(param)
        try:
            for i in range(n_params):
                fusion_net.params['{}/{}'.format(prefix, layer_name)][i].data[...] = param[i].data[...]
        except Exception as e:
            print(e)

fusion_net.save('init_fusion.caffemodel')
