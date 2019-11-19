
This repository is for the review of paper "Overcoming Multi-Model Forgetting in One-Shot NAS with Diversity Maximization" submitted to CVPR 2020


## Requirements
```
Python == 3.6.2, PyTorch == 1.0.0, cuda-9.0, cudnn7.1-9.0

Please download the CIFAR100 dataset in https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz, and save it in the 'data' folder
```


## Pretrained models

* Test on CIFAR10 with the best reported architecture trained with 600 epochs

```
cd CNN && python test.py --auxiliary --model_path ./trained_models/Random_NSAS_CIFAR10_600.pt
```
* Expected result: 2.64% test error rate with 3.08M model params.


* Test on CIFAR10 with the best reported architecture trained with 1000 epochs

```
cd CNN && python test.py --auxiliary --model_path ./trained_models/Random_NSAS_CIFAR10_1000.pt
```
* Expected result: 2.50% test error rate with 3.08M model params.


* Test on CIFAR100 with the best reported architecture
```
cd CNN && python test_100.py --auxiliary --model_path ./trained_models/Random_NSAS_CIFAR100.pt

```
* Expected result: 17.56% top1 test error with 3.13M  model params.


* Test on CIFAR100 with the best reported architecture with 50 initial filters
```
cd CNN && python test_100.py --auxiliary --init_channels 50 --model_path ./trained_models/Random_NSAS_CIFAR100_50F.pt

```
* Expected result: 16.85% top1 test error with 5.8M  model params.



* Test on PTB
```
cd RNN && python test.py --model_path ./trained_models/Random_NSAS_PTB.pt

```
* Expected result: 56.84 test perplexity with 23M model params.



## Architecture search (using small proxy models)
```
Please see the GDAS_NSAS.ipynb, RandomNAS_NSAS.ipynb documents in all folders
```

## Architecture evaluation (using full-sized models)
To evaluate our best cells by training from scratch, run
```
cd CNN && python train.py --auxiliary --cutout            # CIFAR-10
cd CNN && python train_100.py --auxiliary --cutout            # CIFAR-100
cd CNN && python train_100.py --auxiliary --cutout --init_channels 50            # CIFAR-100 with 50 initial chanels
cd RNN && python train.py                                 # PTB
```

## Architecture Visualization
Package graphviz is required to visualize the learned cells, visulize the best reported architectures in this paper
```
cd CNN && python visualize.py Random_NSAS 
cd RNN && python visualize.py Random_NSAS 
```
mmmmmmmmmmmmmmmm
llllllllllll

llllll

