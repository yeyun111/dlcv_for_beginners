## Generative Adversarial Networks (GANs) with 2D Samples
Blog(in Chinese): 
[to be updated](https://zhuanlan.zhihu.com/p/xxxxxx)  
Inspired & based on [Dev Nag's GAN example](https://github.com/devnag/pytorch-generative-adversarial-networks):  
1) Use batch size instead of cardinality to achieve better convergency, the original version is actually generating by default 100(cardinality) dimensional gaussian distribution, so the convergency is **BAD**.   
2) 2D example, with visualizations.  
3) Demo of conditional GAN.  


## Introduction
Play with GAN to generate 2D samples that you can define your own probability density function (PDF) with an gray image.  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/example_z.jpg)  

## 2D Sampling
> python sampler.py

Will demo 10000 samples from the PDF defined by a gray image.  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/test_2d_sampling_batman.png)  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/test_2d_sampling_binary.png)  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/test_2d_sampling_triangle.png)  

## GAN
> python gan_demo.py inputs/zig.jpg  

Training will be visualized as the following:  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_zig.gif)

More examples:  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_Z.gif)  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_circle.gif)  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_random.gif)

## Conditional GAN
For more complex distributions, conditional GAN is much better. This demo reads distributions from different pdfs, encoding conditions as one-hot vector.

> python cgan_demo.py inputs/binary

Training will be visualized as the following:  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/cgan_binary.gif)  
Compared to vanilla GAN version:  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_binary.gif)  

More examples:  
Vortex with C-GAN  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/cgan_vortex.gif)  

Vortex with vanilla GAN  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_vortex.gif)  

Pentagram with C-GAN  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/cgan_penta.gif)  

Pentagram with vanilla GAN  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_penta.gif)  

## Increase latent space dimensionality and model complexity
> python gan_demo.py -h  
> python cgan_demo.py -h

