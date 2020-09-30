# Conditional GAN
Giving a random tensor and a label(0-9), generate an hand-write image. 

[paper](https://arxiv.org/pdf/1411.1784.pdf)
## Dataset
MNIST
## Train
```python
python tools/train.py
```
## Test
```python
python tools/test.py
```
## Loss curve
As we can see from pictures as follows, discriminator loss goes down till epoch 13, then begins to raise till epoch 17. after epoch 17, the train process collapses.
generator loss behaves the oppsite.
![loss.jpg](results/loss.jpg)
## results
![epoch16.jpg](results/epoch16.jpg)
