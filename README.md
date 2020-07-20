
This repository is for the "Overcoming Multi-Model Forgetting in One-Shot NAS With Diversity Maximization" accepted by IEEE CVPR 2020.



## Requirements
```
Python == 3.6.2, PyTorch == 1.0.0, cuda-9.0, cudnn7.1-9.0

Please download the CIFAR100 dataset in https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz, and save it in the 'data' folder
```


## Pretrained models


* Test on CIFAR10 with the best reported architecture with NSAS

```
cd CNN && python test.py --auxiliary --model_path ./trained_models/Random_NSAS_CIFAR10_best.pt --arch Random_NSAS
```
* Expected result: 2.50% test error rate with 3.08M model params.


* Test on CIFAR100 with the best reported architecture with NSAS
```
cd CNN && python test_100.py --auxiliary --model_path ./trained_models/Random_NSAS_CIFAR100.pt --arch Random_NSAS

```
* Expected result: 17.56% top1 test error with 3.13M  model params.



* Test on CIFAR100 with the best reported architecture with NSAS-C
```
cd CNN && python test_100.py --auxiliary --model_path ./trained_models/Random_C_CIFAR100/weights.pt --arch Random_NSAS_C

```
* Expected result: 16.69% top1 test error with 3.59M  model params.



* Test on ImageNET with the best reported architecture with NSAS-C
```
cd CNN && python test_imagenet.py --auxiliary --model_path ./trained_models/Random_NSAS_C_imagenet/model_best.pth.tar --arch  Random_NSAS_C 

```
* Expected result: 25.5% top1 test error with 5.4M  model params.



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
cd CNN && python train_imagenet.py --auxiliary        # imagenet
cd RNN && python train.py                                 # PTB
```

## Architecture Visualization
Package graphviz is required to visualize the learned cells, visulize the best reported architectures in this paper
```
cd CNN && python visualize.py Random_NSAS 
cd CNN && python visualize.py Random_NSAS_C
cd RNN && python visualize.py Random_NSAS 
```
## Codes and Experimental results on NAS-Bench-201
Please find these codes and results in NAS-Bench-201 folder


## Reference
If you use our code in your research, please cite our paper accordingly.
```
@inproceedings{zhang2020overcoming,
  title={Overcoming Multi-Model Forgetting in One-Shot NAS with Diversity Maximization},
  author={Zhang, Miao and Li, Huiqi and Pan, Shirui and Chang, Xiaojun and Su, Steven},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7809--7818},
  year={2020}
}
```



