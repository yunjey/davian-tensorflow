# Deep Convolutional Generative Adverserial Network

TensorFlow reimplementation of [Deep Convolutional Generative Adverserial Network(DCGAN)] (https://arxiv.org/abs/1511.06434) for DAVIAN lab study.

You can download pdf file [here](https://github.com/yunjey/davian-tensorflow/raw/master/notebooks/week4/DCGAN.pdf).

![alt text] (jpg/tensorboard.jpg)

<br>

## Usage

#### Clone the repository
```bash
git clone https://github.com/yunjey/davian-tensorflow.git
cd davian-tensorflow/notebooks/week4
```

#### Download CelebA image dataset
```bash
chmod +x download.sh
./download.sh
```

#### Center crop and resize image to 64x64 
```bash
python prepro.py
```

#### Train the model. 
```bash
python train.py

```

#### Real time debugging
Open the new terminal, run command below and open http://163.152.20.64/:6005/ on your web browser.
```bash
source anaconda2/bin/activate ~/anaconda2   or   source anaconda3/bin/activate ~/anaconda3
cd davian-tensorflow/notebooks/week4
tensorboard --logdir=log --port=6005
```
<br>


## References
Generative Adverserial Network(GAN): https://arxiv.org/abs/1406.2661

Deep Convolutional Generative Adverserial Network(DCGAN): https://arxiv.org/abs/1511.06434

Carpedm20's implementation: https://github.com/carpedm20/DCGAN-tensorflow
