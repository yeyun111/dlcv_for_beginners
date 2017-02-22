## Step 1

> python make_noises.py

## Step 2 

> python gen_label.py

## Step 3

> python gen_hdf5.py train.txt  
> python gen_hdf5.py val.txt

## Step 4

> /path/to/caffe/build/tools/caffe train -solver solver.prototxt

## Step 5
> python predict.py test.txt

## Visualize Conv1 Kernels
> python visualize_conv1_kernels.py
