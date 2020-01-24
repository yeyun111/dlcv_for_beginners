《深度学习与计算机视觉》配套代码  
===
感谢大家在issue里提的各种问题，因为太久没有更新了，本打算在春节前集中精力更新一把，不过看了一圈之后有些无从下手的感觉，再加上和图书公司的编辑讨论过应该不会出第二版或者重新印刷的版本，所以决定放弃……  

我自己还发现的一个错误是关于特征值那块，SVD并不是用于不正定矩阵的，而是用于不对称矩阵的，详见更新后的知乎回答：
https://www.zhihu.com/question/20507061/answer/120540926
其他部分的勘误要么已经在勘误.pdf里，要么在issue里大家的讨论基本上是正确的  

本repo预计不会再更新(坦白讲本书后半部分内容已过时)，如实在有问题请给我本人github私信：yeyun11  

祝各位春节快乐，2020年除夕

===
![封面](https://raw.githubusercontent.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/master/fm.jpg)

原名《深度学习与计算机视觉：实例入门》，请注意：**这本书定位是入门书**。

代码[点这里](https://github.com/frombeijingwithlove/dlcv_for_beginners)。所有彩色图表电子版下载[点这里](https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/tree/master/figs_n_plots)，第五章和第六章的彩色图表参见在线版：[第五章上](https://zhuanlan.zhihu.com/p/24162430)，[第五章下](https://zhuanlan.zhihu.com/p/24309547)，[第六章](https://zhuanlan.zhihu.com/p/24425116)。

因为某些我无法理解的原因，书中英文被出版社要求强行翻译，最后：1）部分英文不同程度的被翻译成了中文，2）导致英文文献占大部分的文献列表未能放到书中。引用文献列表[点这里](https://github.com/frombeijingwithlove/dlcv_for_beginners/blob/master/reference.pdf)。  

内容错误请到[这里](https://github.com/frombeijingwithlove/dlcv_for_beginners/issues)提出。勘误表点[这里](https://github.com/frombeijingwithlove/dlcv_for_beginners/blob/master/errata.pdf)。

购买链接：[京东](https://item.jd.com/12152559.html)，[亚马逊](https://www.amazon.cn/gp/product/B074JWSF99)，[当当](http://product.dangdang.com/25138676.html)

## 代码快速指引
[第五章：numpy、matplotlib可视化的例子](https://github.com/frombeijingwithlove/dlcv_for_beginners/tree/master/chap5)  
[第六章：物体检测标注小工具和本地数据增强小工具](https://github.com/frombeijingwithlove/dlcv_for_beginners/tree/master/chap6)  
[第七章：二维平面分类，分别基于Caffe和MXNet](https://github.com/frombeijingwithlove/dlcv_for_beginners/tree/master/chap7)  
[第八章：MNIST分类，分别基于Caffe和MXNet](https://github.com/frombeijingwithlove/dlcv_for_beginners/tree/master/chap8)  
[第九章：基于Caffe回归图像混乱程度，及卷积核可视化](https://github.com/frombeijingwithlove/dlcv_for_beginners/tree/master/chap9)   
[第十章：从ImageNet预训练模型进行迁移学习美食分类模型，混淆矩阵，ROC曲线绘制及模型类别响应图可视化，基于Caffe](https://github.com/frombeijingwithlove/dlcv_for_beginners/tree/master/chap10)  
[第十二章：用MNIST训练Siamese网络，t-SNE可视化，基于Caffe](https://github.com/frombeijingwithlove/dlcv_for_beginners/tree/master/chap12)  
[书中未包含杂七杂八：包括制造对抗样本(Caffe)、二维GAN及训练过程可视化(PyTorch)、给色情图片自动打马赛克(Caffe)、模型融合(Caffe)、图像分割(PyTorch)](https://github.com/frombeijingwithlove/dlcv_for_beginners/tree/master/random_bonus)  
[模型剪枝(PyTorch)](https://github.com/yeyun11/pytorch-network-slimming)
