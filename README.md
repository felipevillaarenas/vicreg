# VICReg: Variance-Invariance-Covariance Regularization For Self-Supervised Learning



<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7--3.9-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.12+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://colab.research.google.com/drive/1qQB85JVEIml9pNMIeES2hMfFZZkKmB75?usp=sharing"><img alt="Lightning" src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252"></a>



</div>
<br><br>



## 📌&nbsp;&nbsp;Introduction
This repository provides a PyTorch Lighting implementation for VICReg, as described in the paper [VICReg: Variance-Invariance-Covariance Regularization For Self-Supervised Learning](https://arxiv.org/pdf/2105.04906.pdf). This repo is inspired on the original repository of Meta AI. 

> This module was written with the style used in [Lightning Bolts](https://www.pytorchlightning.ai/bolts) for other SOTA Self-Supervised models.


### Why PyTorch Lightning?
[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) is a lightweight PyTorch wrapper for high-performance AI research.
It makes your code neatly organized and provides lots of useful features, like ability to run model on CPU, GPU, multi-GPU cluster and TPU.

### Why Lightning Bolts?
[Lightning Bolts](https://www.pytorchlightning.ai/bolts)  is a community-built deep learning research and production toolbox, featuring a collection of well established and SOTA models and components, pre-trained weights, callbacks, loss functions, data sets, and data modules.​


### How to used this module?
Here are some examples!

**Python**

```python
model = VICReg( arch="resnet18", 
                maxpool1=False, 
                first_conv=False, 
                proj_hidden_dim=2048, 
                proj_output_dim=2048,
                invariance_coeff=25.0,
                variance_coeff=25.0,
                covariance_coeff=1.0,
                optimizer="lars",
                learning_rate=0.3,
                warmup_steps=10)

        dm = CIFAR10DataModule(batch_size=128, num_workers=0)

        dm.train_transforms = VICRegTrainDataTransform(
                input_height=32, 
                gaussian_blur=False, 
                jitter_strength=1.0
                )

        dm.val_transforms = VICRegEvalDataTransform(
                input_height=32, 
                gaussian_blur=False, 
                jitter_strength=1.0
                )

        trainer = pl.Trainer()
        trainer.fit(model, datamodule=dm)
```

**Command line interface** [`cifar10`]

```
python vicreg_module.py 
                --accelerator gpu
                --devices 1
                --dataset cifar10
                --data_dir /path/to/cifar/
                --batch_size 128
                --arch resnet18
                --maxpool1 False
                --first_conv False, 
                --proj_hidden_dim 2048
                --proj_output_dim 2048
                --invariance_coeff 25.0
                --variance_coeff 25.0
                --covariance_coeff 1.0
                --optimizer adam
                --learning_rate 0.3
                --warmup_steps 10

```

**Command line interface** [`imagenet`]
    
```
python vicreg_module.py
                --accelerator gpu
                --devices 1
                --dataset imagenet
                --data_dir /path/to/imagenet/
                --batch_size 512
                --arch resnet50
                --maxpool1 True
                --first_conv True, 
                --proj_hidden_dim 8192
                --proj_output_dim 8192
                --invariance_coeff 25.0
                --variance_coeff 25.0
                --covariance_coeff 1.0
                --optimizer lars
                --learning_rate 0.6
                --warmup_steps 10
```

### Check the Colab version

If you love notebooks and free GPUs, the Colab version of this repository can be found [here](https://colab.research.google.com/drive/1qQB85JVEIml9pNMIeES2hMfFZZkKmB75?usp=sharing)


<br>