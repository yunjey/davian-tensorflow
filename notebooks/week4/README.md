# Deep Convolutional Generative Adverserial Network

![alt text] (jpg/tensorboard.jpg)

<br>

## Usage

#### Download celeb image dataset
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
Open the new terminal, run command below and open http://163.152.51.7/:6005/ on your web browser.
```bash
tensorboard --logdir=log --port=6005
```
<br>


## Reference
Generative Adverserial Network(GAN): https://arxiv.org/abs/1406.2661

Deep Convolutional Generative Adverserial Network(DCGAN): https://arxiv.org/abs/1511.06434

Carpedm20's implementation: https://github.com/carpedm20/DCGAN-tensorflow
