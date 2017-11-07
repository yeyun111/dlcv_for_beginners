## Image Segmentation with Simplified & Customizable U-Net & TriangleNet by PyTorch

### Step 1  
In root folder, create "train" & "val" folder containing two folders with images and segmentations (in lossless format, e.g. PNG). Default folder for images is "image", for segmentations is "segmentations". Or can be specified in config file (see example.cfg)

### Step 2  
> python main.py train [path/to/root_folder] --config [path/to/configfile]

## TriangleNet
Compare image and label at different scale in training, refer to networks.py for more details