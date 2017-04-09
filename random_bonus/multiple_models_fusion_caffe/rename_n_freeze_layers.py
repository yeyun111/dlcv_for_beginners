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
