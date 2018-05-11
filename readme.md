# Awesome Tensorlayer - A curated list of dedicated resources

<a href="https://tensorlayer.readthedocs.io/en/stable/">
<div align="center">
	<img src="https://raw.githubusercontent.com/tensorlayer/tensorlayer/master/img/tl_transparent_logo.png" width="50%" height="30%"/>
</div>
</a>

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![Build Status](https://api.travis-ci.org/tensorlayer/awesome-tensorlayer.svg?branch=master)](https://travis-ci.org/tensorlayer/awesome-tensorlayer)

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

## Contribute

Contributions welcome! Read the [contribution guidelines](contributing.md) first.

## Contents

- [Tutorials - Tips and Tricks](#tutorials---tips-and-tricks)
- [Basics](#basics)
- [Computer Vision](#computer-vision)
- [Natural Language Processing](#natural-language-processing)
- [Reinforcement Learning](#reinforcement-learning)
- [Adversarial Learning](#adversarial-learning)
- [Pretrained Models](#pretrained-models)
- [Miscellaneous](#miscellaneous)
- [Research papers using TensorLayer](#research-papers-using-tensorlayer)

## Tutorials - Tips and Tricks

 - [Tricks to use TL](https://github.com/wagamamaz/tensorlayer-tricks) is a good introduction to start using TensorLayer.


## Basics

#### Data Manipulation

###### Image Preprocessing

- [Image Augmentation](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_image_preprocess.py) randomly augment images with flipped or cropped images.

#### TF Records

- [Read and Save data in TFRecord Format](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_tfrecord.py).

- [Read and Save time-series data in TFRecord Format](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_tfrecord3.py).

- [Convert CIFAR10 in TFRecord Format for performance optimization](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_tfrecord2.py).

#### Loading Data

- [Convolutional Network](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_cifar10.py) working on the dataset CIFAR10 using TensorLayer CIFAR10 data loader.

- [Convolutional Network](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_cifar10_tfrecord.py) working on the dataset CIFAR10 using TFRecords.

#### Multi Layer Perceptron (MLP)

- [Simple MLP Network](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mnist_simple.py) trained on MNIST dataset.

#### Keras

- [Using Keras Layers with Tensorlayer](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_keras.py).

#### TF-Slim

- [Using TF-Slim Layers with Tensorlayer](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_tfslim.py).

- [Using TF-Slim Networks with Tensorlayer](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_inceptionV3_tfslim.py) an example with the CNN InceptionV3 by [[C. Szegedy et al, 2015]](https://arxiv.org/abs/1512.00567).

#### Dropout

- [Using Dropout in Tensorlayer](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mlp_dropout1.py) - Method 1 using *tl.layers.DropoutLayer* and *network.all_drop*.

- [Using Dropout in Tensorlayer](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mlp_dropout2.py) - Method 2 using *tl.layers.DropoutLayer* and *is_train*.

#### Datasets

###### MNIST

- [Downloading and Loading MNIST](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mnist.py) using TensorLayer CIFAR10 data loader.

- [Downloading and Loading MNIST using Docker in Swarm Mode](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mnist_distributed.py) for distributed training.
 
###### PASCAL VOC

- [Downloading and Loading PASCAL VOC with TensorFlow Dataset API](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_tf_dataset_voc.py) using TensorLayer VOC data loader.


## Computer Vision

#### State of the Art Networks


###### VGGNet16

- [VGGNet16 Network](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_vgg16.py) working on the dataset ImageNet using the TFSlim implementation based on the paper by [[K. Simonyan et al, 2014]](https://arxiv.org/abs/1409.1556).

- [VGGNet16 Network](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_vgg16.py) using the Tensorlayer packaged class based on TF-Slim implementation working on the dataset ImageNet using the TFSlim implementation based on the paper by [[K. Simonyan et al, 2014]](https://arxiv.org/abs/1409.1556).


###### VGGNet19

- [VGGNet19 Network](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_vgg19.py) working on the dataset ImageNet using the TFSlim implementation based on the paper by [[K. Simonyan et al, 2014]](https://arxiv.org/abs/1409.1556).


###### InceptionV3

- [InceptionV3 Network](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_inceptionV3_tfslim.py) working on the dataset ImageNet using the TF-Slim implementation based on the paper by [[C. Szegedy et al, 2015]](https://arxiv.org/abs/1512.00567).

- [InceptionV3 Network - Distributed](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_imagenet_inceptionV3_distributed.py) working on the dataset ImageNet based on the paper by [[C. Szegedy et al, 2015]](https://arxiv.org/abs/1512.00567).


#### CNN and Computational Speed or Memory Footprint Bandwitdh Optimization

###### FP8 (float8) and FP16 (float16)

- [Convolutional Network using FP16 (float16)](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mnist_float16.py) on the MNIST dataset.


###### MobileNet

- [MobileNet Network](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mobilenet.py) for mobile vision applications using the dataset ImageNet based on the paper by [[A. G. Howard et al, 2017]](https://arxiv.org/abs/1704.04861).

- [MobileNetV1 Network](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_mobilenetv1.py) using the Tensorlayer packaged class based on TF-Slim implementation for mobile vision applications using the dataset ImageNet based on the paper by [[A. G. Howard et al, 2017]](https://arxiv.org/abs/1704.04861).


###### SqueezeNet

- [SqueezeNet Network](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_squeezenet.py) - a fast and very small (< 0.5MB) network with AlexNet performances using the dataset ImageNet based on the paper by [[F. N. Iandola et al, 2016]](https://arxiv.org/abs/1602.07360).

- [SqueezeNetV1 Network](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_squeezenetv1.py) using the Tensorlayer packaged class based on TF-Slim implementation - a fast and very small (< 0.5MB) network with AlexNet performances using the dataset ImageNet based on the paper by [[F. N. Iandola et al, 2016]](https://arxiv.org/abs/1602.07360).


###### DoReFa

- [Convolutional Network with DoReFa compression](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_dorefanet_mnist_cnn.py) working on the dataset MNIST based on the paper by [[S. Zhou et al, 2016]](https://arxiv.org/abs/1606.06160).

- [Convolutional Network with DoReFa compression](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_dorefanet_cifar10_tfrecord.py) working on the dataset CIFAR10 using TFRecords based on the paper by [[S. Zhou et al, 2016]](https://arxiv.org/abs/1606.06160).


###### Binary Networks

- [Binary Convolutional Network](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_binarynet_mnist_cnn.py) working on the dataset MNIST using TensorLayer MNIST data loader.

- [Binary Convolutional Network](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_binarynet_cifar10_tfrecord.py) working on the dataset CIFAR10 using TFRecords.


###### Ternary Networks

- [Ternary Convolutional Network](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ternaryweight_mnist_cnn.py) working on the dataset MNIST using TensorLayer MNIST data loader.

- [Ternary Convolutional Network](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ternaryweight_cifar10_tfrecord.py) working on the dataset CIFAR10 using TFRecords.

<!-- ==========================

 - ArcFace: Additive Angular Margin Loss for Deep Face Recognition, see [InsignFace](https://github.com/auroua/InsightFace_TF).

 - Wide ResNet (CIFAR) by [ritchieng](https://github.com/ritchieng/wideresnet-tensorlayer).
 
 - [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025) by [zsdonghao](https://github.com/zsdonghao/Spatial-Transformer-Nets).

 - [U-Net for brain tumor segmentation](https://github.com/zsdonghao/u-net-brain-tumor).

 - Variational Autoencoder (VAE) for (CelebA) by [yzwxx](https://github.com/yzwxx/vae-celebA).

 - Variational Autoencoder (VAE) for (MNIST) by [BUPTLdy](https://github.com/BUPTLdy/tl-vae).
 
 - Image Captioning - Reimplementation of Google's [im2txt](https://github.com/tensorflow/models/tree/master/research/im2txt) by [zsdonghao](https://github.com/zsdonghao/Image-Captioning). 
  
 -->

## Natural Language Processing

#### LSTM

- [Text Generation with LSTMs](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_generate_text.py).

- [Predicting the next word with LSTMs](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ptb_lstm.py) on the PTB dataset based on the following [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

- [Predicting the next word with Tuple State LSTMs](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py) on the PTB dataset based on the following [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

#### Embedding Networks

###### FastText

- [FastText Classifier](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_imdb_fasttext.py) running on the IMDB dataset based on the paper by [[A. Joulin et al, 2016]](https://arxiv.org/abs/1607.01759).

###### Word2Vec

- [Minimalistic Implementation of Word2Vec](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_word2vec_basic.py) based on the paper by [[T. Mikolov et al, 2013]](https://arxiv.org/abs/1310.4546).


<!-- =====================

 - Chinese Text Anti-Spam by [pakrchen](https://github.com/pakrchen/text-antispam).
 
 - [Chatbot in 200 lines of code](https://github.com/tensorlayer/seq2seq-chatbot) for [Seq2Seq](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#simple-seq2seq).
 
-->


## Reinforcement Learning

#### Actor Critic

- [Asynchronous Advantage Actor Critic (A3C)](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_bipedalwalker_a3c_continuous_action.py) with Continuous Action Space based on this [blog post](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-3-A3C/).

- [Actor-Critic using TD-error](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_cartpole_ac.py) as the Advantage, Reinforcement Learning based on this [blog post](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-1-actor-critic/).

#### Monte Carlo Methods

- [Monte-Carlo Policy Network](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_atari_pong.py) - Code working with Pong Game on ATARI - Related [blog post](http://karpathy.github.io/2016/05/31/rl/) from Andrej Karpathy.

#### Q-Learning

- [Deep Q Network](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_frozenlake_dqn.py) with Tables and Neural Networks on the FrozenLake OpenAI Gym - Related [blog post](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0).

<!-- ===================== 

 - Asynchronous Policy Gradient using TensorDB (Atari Ping Pong) by [nebulaV](https://github.com/akaraspt/tl_paper).
 
 - [DAGGER](https://www.cs.cmu.edu/%7Esross1/publications/Ross-AIStats11-NoRegret.pdf) for ([Gym Torcs](https://github.com/ugo-nama-kun/gym_torcs)) by [zsdonghao](https://github.com/zsdonghao/Imitation-Learning-Dagger-Torcs).
 
 - [TRPO](https://arxiv.org/abs/1502.05477) for continuous and discrete action space by [jjkke88](https://github.com/jjkke88/RL_toolbox).
 
 -->


## Adversarial Learning

<!-- 
- DCGAN (CelebA). Generating images by [Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) by [zsdonghao](https://github.com/tensorlayer/dcgan).

- [Generative Adversarial Text to Image Synthesis](https://github.com/zsdonghao/text-to-image).

- [Unsupervised Image to Image Translation with Generative Adversarial Networks](https://github.com/zsdonghao/Unsup-Im2Im).

- [Improved CycleGAN](https://github.com/luoxier/CycleGAN_Tensorlayer) with resize-convolution.

- [Super Resolution GAN](https://arxiv.org/abs/1609.04802) by [zsdonghao](https://github.com/tensorlayer/SRGAN).

- [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717) by [2wins](https://github.com/2wins/BEGAN-tensorlayer).

- [DAGAN: Fast Compressed Sensing MRI Reconstruction](https://github.com/nebulaV/DAGAN). 
-->

## Pretrained Models

<!-- 

 - More CNN implementations of [TF-Slim](https://github.com/tensorflow/models/tree/master/research/slim) can be connected to TensorLayer via SlimNetsLayer.

 - All pretrained models in [here](https://github.com/tensorlayer/pretrained-models).

 -->

## Miscellaneous

###### TensorLayer DB: TensorDB

- [What is TensorDB](http://tensorlayer.readthedocs.io/en/latest/modules/db.html).

- TensorDB applied on Pong Game on OpenAI Gym: [Trainer File](https://github.com/akaraspt/tl_paper/blob/master/tutorial_tensordb_atari_pong_trainer.py) and [Generator File](https://github.com/akaraspt/tl_paper/blob/master/tutorial_tensordb_atari_pong_generator.py) based on the following [blog post](http://karpathy.github.io/2016/05/31/rl/).

- TensorDB applied to classification task on MNIST dataset: [Master File](https://github.com/akaraspt/tl_paper/blob/master/tutorial_tensordb_cv_mnist_master.py) and [Worker File](https://github.com/akaraspt/tl_paper/blob/master/tutorial_tensordb_cv_mnist_worker.py).

###### TensorFlask DB: TensorDB

- [TensorFlask](https://github.com/JoelKronander/TensorFlask) - a simple webservice API to process HTTP POST requests using Flask and TensorFlow/Layer.


## Research papers using TensorLayer

- [An example research paper](#) by [A. Author et al, 2018]
