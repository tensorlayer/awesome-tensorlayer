# Awesome Tensorlayer - A curated list of dedicated resources

<a href="https://tensorlayer.readthedocs.io/en/stable/">
<div align="center">
	<img src="https://raw.githubusercontent.com/tensorlayer/tensorlayer/master/img/tl_transparent_logo.png" width="50%" height="30%"/>
</div>
</a>

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

You have just found TensorLayer! High performance DL and RL library for industry and academic.

## Contribute

Contributions welcome! Read the [contribution guidelines](contributing.md) first.

## Contents

- [Tutorials - Tips and Tricks](#tutorials---tips-and-tricks)
- [1. Basics Examples](#basics)
- [2. Computer Vision](#2.-computer-vision)
- [3. Natural Language Processing](#natural-language-processing)
- [4. Reinforcement Learning](#reinforcement-learning)
- [5. Adversarial Learning](#adversarial-learning)
- [6. Pretrained Models](#pretrained-models)
- [7. Auto Encoders](#auto-encoders)
- [Data and Model Managment Tools](#data-and-model-managment-tools)


## Tutorials - Tips and Tricks

 - [Tricks to use TensorLayer](https://github.com/wagamamaz/tensorlayer-tricks) is a third party repository to collect tricks to use TensorLayer better.

## 1. Basics Examples

Get start with TensorLayer.

#### 1.1 MNIST - Hello World

Training MNIST with Dropout is the **Hello World** in deep learning.

- [Using Dropout in Tensorlayer](https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_mlp_dropout1.py) - Method 1 using *DropoutLayer* and *network.all_drop* to switch training and testing.

- [Using Dropout in Tensorlayer](https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_mlp_dropout2.py) - Method 2 using *DropoutLayer* and *is_train* to switch training and testing.

#### 1.2 CIFAR10 - Data Augmentation 

In deep learning, data augmentation is a key fator 

#### Data Manipulation

###### Image Preprocessing

- [Image Augmentation using Python](https://github.com/tensorlayer/tensorlayer/blob/master/examples/data_process/tutorial_image_preprocess.py) randomly augment images with flipped or cropped images.

- [Downloading and Preprocessing PASCAL VOC using TensorFlow Dataset API](https://github.com/tensorlayer/tensorlayer/blob/master/examples/data_process/tutorial_tf_dataset_voc.py) with TensorLayer VOC data loader.


#### TF Records

- [Read and Save data in TFRecord Format](https://github.com/tensorlayer/tensorlayer/blob/master/examples/data_process/tutorial_tfrecord.py).

- [Read and Save time-series data in TFRecord Format](https://github.com/tensorlayer/tensorlayer/blob/master/examples/data_process/tutorial_tfrecord3.py).

- [Convert CIFAR10 in TFRecord Format for performance optimization](https://github.com/tensorlayer/tensorlayer/blob/master/examples/data_process/tutorial_tfrecord2.py).

#### Loading Data

- [Convolutional Network](https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_cifar10_placeholder.py) working on the dataset CIFAR10 using TensorLayer CIFAR10 data loader.

- [Convolutional Network](https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_cifar10_tfrecord.py) working on the dataset CIFAR10 using TFRecords.

#### Multi Layer Perceptron (MLP)

- [Simple MLP Network](https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_mnist_simple.py) trained on MNIST dataset.

#### Keras

- [Using Keras Layers with Tensorlayer](https://github.com/tensorlayer/tensorlayer/blob/master/examples/keras_tfslim/tutorial_keras.py).

#### TF-Slim

- [Using TF-Slim Layers with Tensorlayer](https://github.com/tensorlayer/tensorlayer/blob/master/examples/keras_tfslim/tutorial_tfslim.py).

- [Using TF-Slim Networks with Tensorlayer](https://github.com/tensorlayer/tensorlayer/blob/master/examples/pretrained_cnn/tutorial_inceptionV3_tfslim.py) an example with the CNN InceptionV3 by [[C. Szegedy et al, 2015]](https://arxiv.org/abs/1512.00567).



#### Datasets

###### Process PASCAL VOC

- [Downloading and Preprocessing PASCAL VOC using TensorFlow Dataset API](https://github.com/tensorlayer/tensorlayer/blob/master/examples/data_process/tutorial_tf_dataset_voc.py) with TensorLayer VOC data loader. [知乎文章](https://zhuanlan.zhihu.com/p/31466173)


###### More [here](https://tensorlayer.readthedocs.io/en/latest/modules/files.html#load-dataset-functions)

## 2. Computer Vision

#### 2.1 Computer Vision Applications

- Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization, see [here](https://github.com/tensorlayer/adaptive-style-transfer)

- OpenPose: Real-time multi-person keypoint detection library, see [here](https://github.com/tensorlayer/openpose)

- [InsignFace](https://github.com/auroua/InsightFace_TF) - Additive Angular Margin Loss for Deep Face Recognition

- [Spatial-Transformer-Nets (STN)](https://github.com/zsdonghao/Spatial-Transformer-Nets) trained on MNIST dataset based on the paper by [[M. Jaderberg et al, 2015]](https://arxiv.org/abs/1506.02025).

- [Generative Adversarial Text to Image Synthesis](https://github.com/zsdonghao/text-to-image) on bird and flower dataset.

- [Improved CycleGAN using Resize-Convolution](https://github.com/luoxier/CycleGAN_Tensorlayer).

- [U-Net Brain Tumor Segmentation](https://github.com/zsdonghao/u-net-brain-tumor) trained on BRATS 2017 dataset based on the paper by [[M. Jaderberg et al, 2015]](https://arxiv.org/abs/1705.03820) with some modifications.

- [Image2Text: im2txt](https://github.com/zsdonghao/Image-Captioning) based on the paper by [[O. Vinyals et al, 2016]](https://arxiv.org/abs/1609.06647).

- [DAGAN](https://github.com/nebulaV/DAGAN): Fast Compressed Sensing MRI Reconstruction based on the paper by [[G. Yang et al, 2017]](https://doi.org/10.1109/TMI.2017.2785879). 

- [GAN-CLS for Text to Image Synthesis](https://github.com/zsdonghao/text-to-image) based on the paper by [[S. Reed et al, 2016]](https://arxiv.org/abs/1605.05396)

- [Unsupervised Image-to-Image Translation with Generative Adversarial Networks](https://arxiv.org/pdf/1701.02676), [code](https://github.com/zsdonghao/Unsup-Im2Im)

- More Computer Vision Application can be found in Adversarial Learning Section

#### Pretrained models for ImageNet Classification such as VGG16, VGG19, MobileNet, SqueezeNet, Inception can be found in [tensorlayer/pretrained-models](https://github.com/tensorlayer/pretrained-models) and [examples/pretrained_cnn](https://github.com/tensorlayer/tensorlayer/tree/master/examples/pretrained_cnn)

#### CNN and Computational Speed or Memory Footprint Bandwitdh Optimization

###### FP8 (float8) and FP16 (float16)

- [Convolutional Network using FP16 (float16)](https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_mnist_float16.py) on the MNIST dataset.

###### Quantization Networks

See [examples/quantized_net](https://github.com/tensorlayer/tensorlayer/tree/master/examples/quantized_net).

- [Binary Networks](https://arxiv.org/abs/1602.02830) works on [mnist](https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_binarynet_mnist_cnn.py) and  [cifar10](https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_binarynet_cifar10_tfrecord.py).

- [Ternary Network](https://arxiv.org/abs/1605.04711) works on [mnist](https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_ternaryweight_mnist_cnn.py) and [cifar10](https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_ternaryweight_cifar10_tfrecord.py). 

- [DoReFa-Net](https://arxiv.org/abs/1606.06160) works on [mnist](https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_dorefanet_mnist_cnn.py) and [cifar10](https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_dorefanet_cifar10_tfrecord.py).

- [Quantization For Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) works on [mnist](https://github.com/tensorlayer/blob/master/examples/quantized_net/tutorial_quanconv_mnist.py) and [cifar10](https://github.com/tensorlayer/blob/master/examples/quantized_net/tutorial_quanconv_cifar10.py).

## Natural Language Processing

#### ChatBot
 
- [Seq2Seq Chatbot](https://github.com/tensorlayer/seq2seq-chatbot)  in 200 lines of code for [Seq2Seq](https://tensorlayer.readthedocs.io/en/latest/modules/layers.html#simple-seq2seq).

#### Text Generation

- [Text Generation with LSTMs](https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_generation/tutorial_generate_text.py) - Generating Trump Speech.

- Modelling PennTreebank [code1](https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_ptb/tutorial_ptb_lstm.py) and [code2](https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_ptb/tutorial_ptb_lstm_state_is_tuple.py), see [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

#### FastText

- [FastText Classifier](https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_classification/tutorial_imdb_fasttext.py) running on the IMDB dataset based on the paper by [[A. Joulin et al, 2016]](https://arxiv.org/abs/1607.01759).

#### Word2Vec

- [Minimalistic Implementation of Word2Vec](https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_word_embedding/tutorial_word2vec_basic.py) based on the paper by [[T. Mikolov et al, 2013]](https://arxiv.org/abs/1310.4546).

#### Spam Detection

- [Chinese Spam Detector](https://github.com/pakrchen/text-antispam).


## Reinforcement Learning

#### Actor Critic

- [Asynchronous Advantage Actor Critic (A3C)](https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_bipedalwalker_a3c_continuous_action.py) with Continuous Action Space based on this [blog post](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-3-A3C/).

- [Actor-Critic using TD-error](https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_cartpole_ac.py) as the Advantage, Reinforcement Learning based on this [blog post](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-1-actor-critic/).

#### Policy Network

- [Deep Policy Network](https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_atari_pong.py) - Code working with Pong Game on ATARI - Related [blog post](http://karpathy.github.io/2016/05/31/rl/) from Andrej Karpathy.

#### Q-Learning

- [Deep Q Network](https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_frozenlake_dqn.py) with Tables and Neural Networks on the FrozenLake OpenAI Gym - Related [blog post](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0).

#### Imitation Learning

 - [DAGGER](https://www.cs.cmu.edu/%7Esross1/publications/Ross-AIStats11-NoRegret.pdf) for ([Gym Torcs](https://github.com/ugo-nama-kun/gym_torcs)) by [zsdonghao](https://github.com/zsdonghao/Imitation-Learning-Dagger-Torcs).

#### RL Toolbox
 
 - [RL Toolbox](https://github.com/jjkke88/RL_toolbox) is a reinfore learning tool box, contains TRPO, A3C for ontinous action space by [jjkke88](https://github.com/jjkke88).


## Adversarial Learning

- [SRGAN](https://github.com/tensorlayer/srgan) - A Super Resolution GAN based on the paper by [[C. Ledig et al, 2016]](https://arxiv.org/abs/1609.04802).

- [DCGAN](https://github.com/tensorlayer/dcgan) trained on the CelebA dataset based on the paper by [[A. Radford et al, 2015]](https://arxiv.org/abs/1511.06434).

- [CycleGAN](https://github.com/luoxier/CycleGAN_Tensorlayer) improved with resize-convolution based on the paper by [[J. Zhu et al, 2017]](https://arxiv.org/abs/1703.10593).

- [DAGAN](https://github.com/nebulaV/DAGAN): Fast Compressed Sensing MRI Reconstruction based on the paper by [[G. Yang et al, 2017]](https://doi.org/10.1109/TMI.2017.2785879). 

- [GAN-CLS for Text to Image Synthesis](https://github.com/zsdonghao/text-to-image) based on the paper by [[S. Reed et al, 2016]](https://arxiv.org/abs/1605.05396)

- [Unsupervised Image-to-Image Translation with Generative Adversarial Networks](https://arxiv.org/pdf/1701.02676), [code](https://github.com/zsdonghao/Unsup-Im2Im)

- [BEGAN](https://github.com/2wins/BEGAN-tensorlayer): Boundary Equilibrium Generative Adversarial Networks based on the paper by [[D. Berthelot et al, 2017]](https://arxiv.org/abs/1703.10717).

## Pretrained Models

- All pretrained models can be found [here](https://github.com/tensorlayer/pretrained-models).
- [Tricks to use TensorLayer](https://github.com/wagamamaz/tensorlayer-tricks) provides useful examples.

## Auto Encoders

#### Variational Autoencoder (VAE)

- [Variational Autoencoder](https://github.com/yzwxx/vae-celebA) trained on the CelebA dataset.

- [Variational Autoencoder](https://github.com/BUPTLdy/tl-vae) trained on the MNIST dataset.


## Data and Model Managment Tools

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


