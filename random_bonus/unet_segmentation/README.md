## Simplified & Customizable U-Net Implemenation by PyTorch

### Step 1  
In root folder, create "train" & "val" folder containing two folders with images and segmentations (in lossless format, e.g. PNG). Default folder for images is "image", for segmentations is "segmentations". Or can be specified in config file (see example.cfg)

### Step 2  
> python main.py train [path/to/root_folder] --config [path/to/configfile]

### Human Parsing Example
humanparsing-256.cfg
https://github.com/lemondan/HumanParsing-Dataset