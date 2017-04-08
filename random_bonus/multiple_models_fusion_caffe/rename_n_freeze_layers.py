import sys
import re

layer_name_regex = re.compile('name:\s*"(.*?)"')
lr_mult_regex = re.compile('lr_mult:\s*\d+\.*\d*')

input_filepath = sys.argv[1]
output_filepath = sys.argv[2]
prefix = sys.argv[3]

with open(input_filepath, 'r') as fr, open(output_filepath, 'w') as fw:
    prototxt = fr.read()
    layer_names = set(layer_name_regex.findall(prototxt))
    for layer_name in layer_names:
        prototxt = prototxt.replace(layer_name, '{}/{}'.format(prefix, layer_name))

    lr_mult_statements = set(lr_mult_regex.findall(prototxt))
    for lr_mult_statement in lr_mult_statements:
        prototxt = prototxt.replace(lr_mult_statement, 'lr_mult: 0')

    fw.write(prototxt)

'''
model_def_list = [
    ('mnist_even', 'lenet_even_train_val.prototxt'),
    ('mnist_odd', 'lenet_odd_train_val.prototxt'),
]

fused_prototxt = ''

for prefix, model_def in model_def_list:
    with open(model_def, 'r') as f:
        def_prototxt = f.read()
        layer_names = layer_name_regex.findall(def_prototxt)



with open('lenet_even_train_val.prototxt', 'r') as f:



original_deploy_prototxt = 'deploy.prototxt'
target_train_prototxt = 'fuse_voting_train.prototxt'
ps_weights = 'ps_res50_1by2_all_iter_119000.caffemodel'
pn_weights = 'pn_res50_1by2_all_iter_106000.caffemodel'
sn_weights = 'sn_res50_1by2_all_iter_89000.caffemodel'
mean_file = 'mean.txt'

init_net = caffe.Net(target_train_prototxt, caffe.TEST)

def get_layer_names(prototxt):
    layer_name_regex = 
    with open(prototxt, 'r') as f:
        net_def = f.read()
    return layer_name_regex.findall(net_def)

layer_names = get_layer_names(original_deploy_prototxt)

for bin_type, model_weights in [
    ('ps', ps_weights),
    ('pn', pn_weights),
    ('sn', sn_weights)
    ]:

    cur_net = caffe.Net(original_deploy_prototxt, model_weights, caffe.TEST)
    for layer_name, param in cur_net.params.iteritems():
        print(bin_type, layer_name)
        n_params = len(param)
        try:
            for i in range(n_params):
                init_net.params['{}/{}'.format(bin_type, layer_name)][i].data[...] = param[i].data[...]
        except Exception as e:
            print("WTF", e)

init_net.save('fuse_voting.caffemodel')
'''
