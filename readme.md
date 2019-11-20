# Awesome Tensorlayer - A curated list of dedicated resources

<a href="https://tensorlayer.readthedocs.io/en/stable/">
<div align="center">
	<img src="https://raw.githubusercontent.com/tensorlayer/tensorlayer/master/img/tl_transparent_logo.png" width="50%" height="30%"/>
</div>
</a>

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![Build Status](https://api.travis-ci.org/tensorlayer/awesome-tensorlayer.svg?branch=master)](https://travis-ci.org/tensorlayer/awesome-tensorlayer)

You have just found TensorLayer! High performance DL and RL library for industry and academic.

## Contribute

Contributions welcome! Read the [contribution guidelines](contributing.md) first.

## Contents

- [1. Basics Examples](#1-basics-examples)
- [2. Computer Vision](#2-computer-vision)
- [3. Natural Language Processing](#3-natural-language-processing)
- [4. Reinforcement Learning](#4-reinforcement-learning)
- [5. Adversarial Learning](#5-adversarial-learning)
- [6. Pretrained Models](#6-pretrained-models)
- [7. Auto Encoders](#7-auto-encoders)
- [8. Data and Model Managment Tools](#8-data-and-model-managment-tools)


## 1. Basics Examples

### 1.1 MNIST and CIFAR10 

TensorLayer has two types of models.
Static model allows you to build model in a fluent way while dynamic model allows you to fully control the forward process.
Please read this [DOCS](https://tensorlayer.readthedocs.io/en/latest/user/get_start_model.html#) before you start.

- [MNIST Simplest Example](https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_mnist_simple.py)
- [MNIST Static Example](https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_mnist_mlp_static.py)
- [MNIST Static Example for Reused Model](https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_mnist_mlp_static_2.py)
- [MNIST Dynamic Example](https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_mnist_mlp_dynamic.py)
- [MNIST Dynamic Example for Seperated Models](https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_mnist_mlp_dynamic_2.py)
- [MNIST Static Siamese Model Example](https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_mnist_siamese.py)
- [CIFAR10 Static Example with Data Augmentation](https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_cifar10_cnn_static.py)

### 1.2 DatasetAPI and TFRecord Examples

- [Downloading and Preprocessing PASCAL VOC](https://github.com/tensorlayer/tensorlayer/blob/master/examples/data_process/tutorial_tf_dataset_voc.py) with TensorLayer VOC data loader. [知乎文章](https://zhuanlan.zhihu.com/p/31466173)
- [Read and Save data in TFRecord Format](https://github.com/tensorlayer/tensorlayer/blob/master/examples/data_process/tutorial_tfrecord.py).
- [Read and Save time-series data in TFRecord Format](https://github.com/tensorlayer/tensorlayer/blob/master/examples/data_process/tutorial_tfrecord3.py).
- [Convert CIFAR10 in TFRecord Format for performance optimization](https://github.com/tensorlayer/tensorlayer/blob/master/examples/data_process/tutorial_tfrecord2.py).
- More dataset loader can be found in [tl.files.load_xxx](https://tensorlayer.readthedocs.io/en/latest/modules/files.html#load-dataset-functions)

## 2. Computer Vision

### 2.1 Computer Vision Applications

- [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://github.com/tensorlayer/adaptive-style-transfer)
- [OpenPose: Real-time multi-person keypoint detection](https://github.com/tensorlayer/openpose-plus)
- [InsignFace](https://github.com/auroua/InsightFace_TF) - Additive Angular Margin Loss for Deep Face Recognition
- [Spatial-Transformer-Nets (STN)](https://github.com/zsdonghao/Spatial-Transformer-Nets) trained on MNIST dataset based on the paper by [[M. Jaderberg et al, 2015]](https://arxiv.org/abs/1506.02025).
- [U-Net Brain Tumor Segmentation](https://github.com/zsdonghao/u-net-brain-tumor) trained on BRATS 2017 dataset based on the paper by [[M. Jaderberg et al, 2015]](https://arxiv.org/abs/1705.03820) with some modifications.
- [Image2Text: im2txt](https://github.com/zsdonghao/Image-Captioning) based on the paper by [[O. Vinyals et al, 2016]](https://arxiv.org/abs/1609.06647).
- More Computer Vision Application can be found in [Adversarial Learning Section](#5-adversarial-learning)

### 2.2 CNN and Computational Speed or Memory Footprint Bandwitdh Optimization

#### Quantization Networks

See [examples/quantized_net](https://github.com/tensorlayer/tensorlayer/tree/master/examples/quantized_net).

- [Binary Networks](https://arxiv.org/abs/1602.02830) works on [mnist](https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_binarynet_mnist_cnn.py) and  [cifar10](https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_binarynet_cifar10_tfrecord.py).
- [Ternary Network](https://arxiv.org/abs/1605.04711) works on [mnist](https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_ternaryweight_mnist_cnn.py) and [cifar10](https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_ternaryweight_cifar10_tfrecord.py).
- [DoReFa-Net](https://arxiv.org/abs/1606.06160) works on [mnist](https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_dorefanet_mnist_cnn.py) and [cifar10](https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_dorefanet_cifar10_tfrecord.py).
- [Quantization For Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) works on [mnist](https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_quanconv_mnist.py) and [cifar10](https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_quanconv_cifar10.py).


## 3. Natural Language Processing

### 3.1 ChatBot

- [Seq2Seq Chatbot](https://github.com/tensorlayer/seq2seq-chatbot)  in 200 lines of code for [Seq2Seq](https://tensorlayer.readthedocs.io/en/latest/modules/layers.html#simple-seq2seq).

### 3.2 Text Generation

- [Text Generation with LSTMs](https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_generation/tutorial_generate_text.py) - Generating Trump Speech.
- Modelling PennTreebank [code1](https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_ptb/tutorial_ptb_lstm.py) and [code2](https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_ptb/tutorial_ptb_lstm_state_is_tuple.py), see [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

### 3.3 Text Classification

- [FastText Classifier](https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_classification/tutorial_imdb_fasttext.py) running on the IMDB dataset based on the paper by [[A. Joulin et al, 2016]](https://arxiv.org/abs/1607.01759).

### 3.4 Word Embedding

- [Minimalistic Implementation of Word2Vec](https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_word_embedding/tutorial_word2vec_basic.py) based on the paper by [[T. Mikolov et al, 2013]](https://arxiv.org/abs/1310.4546).

### 3.5 Spam Detection

- [Chinese Spam Detector](https://github.com/pakrchen/text-antispam).


## 4. Reinforcement Learning

- [DRL Tutorial for Academic](https://github.com/tensorlayer/tensorlayer/tree/master/examples/reinforcement_learning)
- [DRL Zoo for Industry](https://github.com/tensorlayer/RLzoo)

## 5. Adversarial Learning

- [DCGAN](https://github.com/tensorlayer/dcgan) trained on the CelebA dataset based on the paper by [[A. Radford et al, 2015]](https://arxiv.org/abs/1511.06434).
- [CycleGAN](https://github.com/tensorlayer/cyclegan) improved with resize-convolution based on the paper by [[J. Zhu et al, 2017]](https://arxiv.org/abs/1703.10593).
- [SRGAN](https://github.com/tensorlayer/srgan) - A Super Resolution GAN based on the paper by [[C. Ledig et al, 2016]](https://arxiv.org/abs/1609.04802).
- [DAGAN](https://github.com/nebulaV/DAGAN): Fast Compressed Sensing MRI Reconstruction based on the paper by [[G. Yang et al, 2017]](https://doi.org/10.1109/TMI.2017.2785879).
- [GAN-CLS for Text to Image Synthesis](https://github.com/zsdonghao/text-to-image) based on the paper by [[S. Reed et al, 2016]](https://arxiv.org/abs/1605.05396)
- [Unsupervised Image-to-Image Translation with Generative Adversarial Networks](https://arxiv.org/abs/1701.02676), [code](https://github.com/zsdonghao/Unsup-Im2Im)
- [BEGAN](https://github.com/2wins/BEGAN-tensorlayer): Boundary Equilibrium Generative Adversarial Networks based on the paper by [[D. Berthelot et al, 2017]](https://arxiv.org/abs/1703.10717).


## 6. Pretrained Models

- The guideline of using pretrained models is [here](https://tensorlayer.readthedocs.io/en/latest/user/get_start_advance.html#pre-trained-cnn).

## 7. Auto Encoders

### Variational Autoencoder (VAE)

- [Variational Autoencoder](https://github.com/yzwxx/vae-celebA) trained on the CelebA dataset.
- [Variational Autoencoder](https://github.com/BUPTLdy/tl-vae) trained on the MNIST dataset.


## 8. Data and Model Managment Tools

- [Why Database?](https://tensorlayer.readthedocs.io/en/stable/modules/db.html).
- Put Tasks into Database and Execute on Other Agents, see [code](https://github.com/tensorlayer/tensorlayer/tree/master/examples/database).
- TensorDB applied on Pong Game on OpenAI Gym: [Trainer File](https://github.com/akaraspt/tl_paper/blob/master/tutorial_tensordb_atari_pong_trainer.py) and [Generator File](https://github.com/akaraspt/tl_paper/blob/master/tutorial_tensordb_atari_pong_generator.py) based on the following [blog post](http://karpathy.github.io/2016/05/31/rl/).
- TensorDB applied to classification task on MNIST dataset: [Master File](https://github.com/akaraspt/tl_paper/blob/master/tutorial_tensordb_cv_mnist_master.py) and [Worker File](https://github.com/akaraspt/tl_paper/blob/master/tutorial_tensordb_cv_mnist_worker.py).


## How to cite TL in Research Papers ?
If you find this project useful, we would be grateful if you cite the TensorLayer paper：

```
@article{tensorlayer2017,
    author  = {Dong, Hao and Supratak, Akara and Mai, Luo and Liu, Fangde and Oehmichen, Axel and Yu, Simiao and Guo, Yike},
    journal = {ACM Multimedia},
    title   = {{TensorLayer: A Versatile Library for Efficient Deep Learning Development}},
    url     = {http://tensorlayer.org},
    year    = {2017}
}
```


# **ENJOY**
