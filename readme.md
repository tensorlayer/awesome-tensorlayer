# Awesome Tensorlayer - A curated list of dedicated resources

<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="https://github.com/tensorlayer/tensorlayer/raw/master/img/tl_transparent_logo.png" width="50%" height="30%"/>
</div>
</a>

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![Build Status](https://travis-ci.org/tensorlayer/awesome-tensorlayer.svg?branch=master)](https://travis-ci.org/tensorlayer/awesome-tensorlayer)

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

- [Tutorials](#tutorials)
- [Basics](#basics)
- [Vision](#vision)
- [Adversarial Learning](#adversarial-learning)
- [Natural Language Processing](#natural-language-processing)
- [Reinforcement Learning](#reinforcement-learning)
- [Pretrained Models](#pretrained-models)
- [Miscellaneous](#miscellaneous)
- [Research papers using TensorLayer](#research-papers-using-tensorLayer)


### Tutorials

Examples can be found [in this folder](https://github.com/zsdonghao/tensorlayer/tree/master/example) and [Github topic](https://github.com/search?q=topic%3Atensorlayer&type=Repositories).

### Basics
 - Multi-layer perceptron (MNIST) - Classification task, see [tutorial\_mnist\_simple.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist_simple.py).
 - Multi-layer perceptron (MNIST) - Classification with dropout using iterator, see [method1](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mlp_dropout1.py) (**use placeholder**) and [method2](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mlp_dropout2.py) (**use reuse**).
 - Denoising Autoencoder (MNIST). Classification task, see [tutorial_mnist.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist.py).
 - Stacked Denoising Autoencoder and Fine-Tuning (MNIST). Classification task, see [tutorial_mnist.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist.py).
 - Convolutional Network (MNIST). Classification task, see [tutorial_mnist.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist.py).
 - Convolutional Network (CIFAR-10). Classification task, see [tutorial\_cifar10.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_cifar10.py) and [tutorial\_cifar10_tfrecord.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_cifar10_tfrecord.py).
 - TensorFlow dataset API for object detection see [here](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_tf_dataset_voc.py).
 - Merge TF-Slim into TensorLayer. [tutorial\_inceptionV3_tfslim.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_inceptionV3_tfslim.py).
 - Merge Keras into TensorLayer. [tutorial_keras.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_keras.py).
 - Data augmentation with TFRecord. Effective way to load and pre-process data, see [tutorial_tfrecord*.py](https://github.com/zsdonghao/tensorlayer/tree/master/example) and [tutorial\_cifar10_tfrecord.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_cifar10_tfrecord.py).
 - Data augmentation with TensorLayer, see [tutorial\_image_preprocess.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_image_preprocess.py).
 - Float 16 half-precision model, see [tutorial\_mnist_float16.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist_float16.py)
 - Distributed Training. [mnist](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist_distributed.py) and [imagenet](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_imagenet_inceptionV3_distributed.py) by [jorgemf](https://github.com/jorgemf).

### Vision
 - ArcFace: Additive Angular Margin Loss for Deep Face Recognition, see [InsignFace](https://github.com/auroua/InsightFace_TF).
 - BinaryNet. Model acceleration, see [mnist](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_binarynet_mnist_cnn.py) [cifar10](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_binarynet_cifar10_tfrecord.py).
 - Ternary Weight Network. Model acceleration, see [mnist](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ternaryweight_mnist_cnn.py) [cifar10](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ternaryweight_cifar10_tfrecord.py).
 - DoReFa-Net. Model acceleration, see [mnist](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_dorefanet_mnist_cnn.py) [cifar10](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_dorefanet_cifar10_tfrecord.py).
 - Wide ResNet (CIFAR) by [ritchieng](https://github.com/ritchieng/wideresnet-tensorlayer).
 - [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025) by [zsdonghao](https://github.com/zsdonghao/Spatial-Transformer-Nets).
 - [U-Net for brain tumor segmentation](https://github.com/zsdonghao/u-net-brain-tumor) by [zsdonghao](https://github.com/zsdonghao/u-net-brain-tumor).
 - Variational Autoencoder (VAE) for (CelebA) by [yzwxx](https://github.com/yzwxx/vae-celebA).
 - Variational Autoencoder (VAE) for (MNIST) by [BUPTLdy](https://github.com/BUPTLdy/tl-vae).
 - Image Captioning - Reimplementation of Google's [im2txt](https://github.com/tensorflow/models/tree/master/research/im2txt) by [zsdonghao](https://github.com/zsdonghao/Image-Captioning).


### Adversarial Learning
- DCGAN (CelebA). Generating images by [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434) by [zsdonghao](https://github.com/zsdonghao/dcgan).
- [Generative Adversarial Text to Image Synthesis](https://github.com/zsdonghao/text-to-image) by [zsdonghao](https://github.com/zsdonghao/text-to-image).
- [Unsupervised Image to Image Translation with Generative Adversarial Networks](https://github.com/zsdonghao/Unsup-Im2Im) by [zsdonghao](https://github.com/zsdonghao/Unsup-Im2Im).
- [Improved CycleGAN](https://github.com/luoxier/CycleGAN_Tensorlayer) with resize-convolution by [luoxier](https://github.com/luoxier/CycleGAN_Tensorlayer).
- [Super Resolution GAN](https://arxiv.org/abs/1609.04802) by [zsdonghao](https://github.com/zsdonghao/SRGAN).
- [BEGAN: Boundary Equilibrium Generative Adversarial Networks](http://arxiv.org/abs/1703.10717) by [2wins](https://github.com/2wins/BEGAN-tensorlayer).
- [DAGAN: Fast Compressed Sensing MRI Reconstruction](https://github.com/nebulaV/DAGAN) by [nebulaV](https://github.com/nebulaV/DAGAN).

### Natural Language Processing
 - Recurrent Neural Network (LSTM). Apply multiple LSTM to PTB dataset for language modeling, see [tutorial_ptb_lstm.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm.py) and [tutorial\_ptb\_lstm\_state\_is_tuple.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py).
 - Word Embedding (Word2vec). Train a word embedding matrix, see [tutorial\_word2vec_basic.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial\_word2vec_basic.py).
 - Restore Embedding matrix. Restore a pre-train embedding matrix, see [tutorial\_generate_text.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_generate_text.py).
 - Text Generation. Generates new text scripts, using LSTM network, see [tutorial\_generate_text.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_generate_text.py).
 - Chinese Text Anti-Spam by [pakrchen](https://github.com/pakrchen/text-antispam).
 - [Chatbot in 200 lines of code](https://github.com/zsdonghao/seq2seq-chatbot) for [Seq2Seq](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#simple-seq2seq).
 - FastText Sentence Classification (IMDB), see [tutorial\_imdb\_fasttext.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_imdb_fasttext.py) by [tomtung](https://github.com/tomtung).


### Reinforcement Learning
 - Policy Gradient / Network (Atari Ping Pong), see [tutorial\_atari_pong.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_atari_pong.py).
 - Deep Q-Network (Frozen lake), see [tutorial\_frozenlake_dqn.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_frozenlake_dqn.py).
 - Q-Table learning algorithm (Frozen lake), see [tutorial\_frozenlake\_q_table.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_frozenlake_q_table.py).
 - Asynchronous Policy Gradient using TensorDB (Atari Ping Pong) by [nebulaV](https://github.com/akaraspt/tl_paper).
 - AC for discrete action space (Cartpole), see [tutorial\_cartpole_ac.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_cartpole_ac.py).
 - A3C for continuous action space (Bipedal Walker), see [tutorial\_bipedalwalker_a3c*.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_bipedalwalker_a3c_continuous_action.py).
 - [DAGGER](https://www.cs.cmu.edu/%7Esross1/publications/Ross-AIStats11-NoRegret.pdf) for ([Gym Torcs](https://github.com/ugo-nama-kun/gym_torcs)) by [zsdonghao](https://github.com/zsdonghao/Imitation-Learning-Dagger-Torcs).
 - [TRPO](https://arxiv.org/abs/1502.05477) for continuous and discrete action space by [jjkke88](https://github.com/jjkke88/RL_toolbox).


### Pretrained Models
 - VGG 16 (ImageNet). Classification task, see [tl.models.VGG16](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_models_vgg16.py) or [tutorial_vgg16.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_vgg16.py).
 - VGG 19 (ImageNet). Classification task, see [tutorial_vgg19.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_vgg19.py).
 - InceptionV3 (ImageNet). Classification task, see [tutorial\_inceptionV3_tfslim.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_inceptionV3_tfslim.py).
 - SqueezeNet (ImageNet). Model acceleration, see [tl.models.SqueezeNetV1](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_models_squeezenetv1.py) or [tutorial_squeezenet.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_squeezenet.py)
 - MobileNet (ImageNet). Model acceleration, see [tl.models.MobileNetV1](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_mobilenetv1.py) or [tutorial_mobilenet.py](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mobilenet.py).
 - More CNN implementations of [TF-Slim](https://github.com/tensorflow/models/tree/master/research/slim) can be connected to TensorLayer via SlimNetsLayer.
 - All pretrained models in [here](https://github.com/tensorlayer/pretrained-models).

### Miscellaneous
 - TensorDB by [fangde](https://github.com/fangde) see [here](https://github.com/akaraspt/tl_paper).
 - A simple web service - [TensorFlask](https://github.com/JoelKronander/TensorFlask) by [JoelKronander](https://github.com/JoelKronander).

### Research papers using TensorLayer
 - [MyPaper](http://www.example.com)
