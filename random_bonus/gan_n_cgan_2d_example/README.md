## Generative Adversarial Networks (GANs) with 2D Samples
Blog(in Chinese): 
[用GAN生成二维样本的小例子](https://zhuanlan.zhihu.com/p/27343585)  
Inspired & based on [Dev Nag's GAN example](https://github.com/devnag/pytorch-generative-adversarial-networks):  
1) Use batch size instead of cardinality to achieve better convergency, the original version is actually generating 100 (cardinality by default) dimensional gaussian distribution for discriminator, so the convergency is **BAD**.   
2) Use 2D samples, with visualization of training.  
3) Demo of conditional GAN.  
4) GPU support. 


## Introduction
Play with GAN to generate 2D samples that you can define your own probability density function (PDF) with a gray image.  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_n_cgan_2d/example_z.jpg)  

## 2D Sampling
> python sampler.py

Will demo 10000 samples from the PDF defined by a gray image.  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_n_cgan_2d/test_2d_sampling_batman.png)  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_n_cgan_2d/test_2d_sampling_binary.png)  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_n_cgan_2d/test_2d_sampling_triangle.png)  

## GAN
> python gan_demo.py inputs/zig.jpg  

Training will be visualized as the following:  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_n_cgan_2d/gan_zig.gif)

More examples:  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_n_cgan_2d/gan_Z.gif)  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_n_cgan_2d/gan_triangle.gif)  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_n_cgan_2d/gan_circle.gif)  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_n_cgan_2d/gan_random.gif)

## Conditional GAN
For more complex distributions, conditional GAN is much better. This demo reads distributions from different pdfs, encoding conditions as one-hot vector.

> python cgan_demo.py inputs/binary

Training will be visualized as the following:  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_n_cgan_2d/cgan_binary.gif)  
Compared to vanilla GAN version:  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_n_cgan_2d/gan_binary.gif)  

More examples:  
Vortex with C-GAN  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_n_cgan_2d/cgan_vortex.gif)  

Vortex with vanilla GAN  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_n_cgan_2d/gan_vortex.gif)  

Pentagram with C-GAN  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_n_cgan_2d/cgan_penta.gif)  

Pentagram with vanilla GAN  
![image](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/random_bonus/gan_n_cgan_2d/gan_penta.gif)  

## Latent space dimensionality / model complexity / learning rates / ...
> python gan_demo.py -h  

