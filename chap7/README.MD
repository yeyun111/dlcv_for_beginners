## Prepare Data
run gen_data.py to generate data in pickle

## MXNet
run simple_mlp.py in mxnet directory to train & visualize result

## Caffe
### step 1
run gen_hdf5.py to convert data to hdf5 format
### step 2
run simple_mlp_train.py to train the model
### step 3
run simple_mlp_test.py to do inference & visualize the result