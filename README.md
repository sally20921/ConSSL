# Contrastive Self-Supervised Learning SOTA Models


 <a href="https://sites.google.com/snu.ac.kr/serileeproject00/projects/">Website</a> 
![PyPI Status](https://badge.fury.io/py/ConSSL.svg)

This repository houses a collection of all self-supervised learning models.

I implemented most of the current state-of-the-art self-supervised learning methods including SimCLRv2, BYOL, SimSiam, MoCov2, and SwAV.

## Install Package
`
pip install ConSSL
`

## Usage
```python
import torch
from ConSSL.self_supervised import SimSiam
from ConSSL.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDatatTransform
from torchvision import models

train_dataset = MyDataset(transform=SimCLRTrainDataTransform())
val_dataset = MyDataset(transforms=SimCLREvalDataTransform())

model = SimSiam()
trainer = Trainer(gpu=4)
trainer.fit(
	model,
	DataLoader(train_dataset),
	DataLoader(val_dataset),
)
```

## Notes

- I found that using SimCLR augmentation directly will sometimes cause the model to collpase. This maybe due to the fact that SimCLR augmentation is too strong.
- Adopting the MoCo augmentation during the warmup stage helps.

## Dataset

Collection of useful datasets including STL10, MNIST, CIFAR10, CIFAR100, ImageNet.

The dataset will be downloaded and is placed in this hierarchy below.

Download imagenet dataset and place it accordingly since ImageNet dataset is too big of a file to download it on code.

```
data/
  imagenet/
    train/
      ...
      n021015556/
        ..
        n021015556_12124.jpeg
	..
      n021015557/
      ...
    val/
      ...
      n021015556/
        ...
	ILSVRC2012_val_00003054.jpeg
	...
      n021015557/
      ...
```

## Stages
### Pretraining (Data Modules)
Data Modules (introduced in PyTorch Lightning 0.9.0) decouple the data from a model. 

A Data Module is simply a collection of a training dataloader, val dataloader and test dataloader. It specifies how to 
- download/prepare data
- train/val/test splits
- transform

You can use it like this.
```python
dm = MNISTDataModule('path/to/data')
model = LiteModel()

trainer = Trainer()
trainer.fit(model, datamodule=dm)
```
You can also use it manually.
```python
dm = MNISTDataModule('/path/to/data')
for batch in dm.train_dataloader():
	...
for batch in dm.val_dataloader():
	...
for batch in dm.test_dataloader():
	...
```
### Contrastive Self-Supervised Learning Models
#### SimCLRv2
##### Usage 
```python
import pytorch_lightning as pl
from ConSSL.models.self_supervised import SimCLRv2
from ConSSL.datamodules import CIFAR10DatatModule
from ConSSL.models.self_supervised.simclr.transforms import (SimCLREvalDataTransform, SimCLRTrainDataTransform)

# data
dm = CIFAR10DataModule(num_workers=0)
dm.train_transforms = SimCLRTrainDataTransform(32)
dm.val_transforms = SimCLREvalDataTransform(32)

# model
model = SimCLRv2(num_samples=dm.num_samples, batch_size=dm.batch_size, dataset='cifar10')

# fit 
trainer = pl.Trainer()
trainer.fit(model, datamodule=dm)

```
##### Results
|Implementation| Dataset     | Architecture | Optimizer|Batch size | Epochs | Linear Evaluation| 
|--------------| ------------| ------------ | ---------|-----------| ------ | -----------------|
|   Original   | CIFAR10     | ResNet50     | LARS     |1024       | 500    | 0.94             | 
|   Mine       | CIFAR10     | ResNet50     | LARS-SGD |1024       | 500    | 0.88             | 

#### to reproduce
```
cd code
# change the configuration setting in config.py
python cli.py pretrain
python cli.py linear_evaluation
```

#### MoCov2
```python
```
#### BYOL
```python
```
#### SwAV
```python
```
##### Results
|Implementation| Dataset     | Architecture | Optimizer|Batch size | Epochs | Linear Evaluation| 
|--------------| ------------| ------------ | ---------|-----------| ------ | -----------------|
|   Original   | CIFAR10     | ResNet50     | LARS     |1024       | 500    | 0.75             | 
|   Mine       | CIFAR10     | ResNet50     | LARS-SGD |1024       | 500    | 0.74             | 

#### SimSiam
```python
```

### Linear Evaluation Protocol
### Semi-Supervised Learning
use imagenet subset from https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/image_classification

### Transfer Learning
## Dependency
- I use latest version of python 3 and python2 is not supported. 
- I use latest version of PyTorch, though tensorflow-gpu is necessary to launch tensorboard.
## Install
```
git clone --recurse-submodules (this repo)
cd $REPO_NAME/code
(use python >= 3.5)
pip install -r requirements.txt
```

### When using docker

build & push & run
```
sudo ./setup-docker.sh
```
directory structure
```
/home/
 /code/
 /data/
```

## Data Folder Structure
```
code/
 cli.py : executable check_dataloading, training, evaluating script
 config.py: default configs
 ckpt.py: checkpoint saving & loading
 train.py : training python configuration file
 evaluate.py : evaluating python configuration file
 infer.py : make submission from checkpoint
 logger.py: tensorboard and commandline logger for scalars
 utils.py : other helper modules
 dataloader/ : module provides data loaders and various transformers
  load_dataset.py: dataloader for classification
  vision.py: image loading helper
 loss/ 
 metric/ : accuracy and loss logging 
 optimizer/
 ...
data/
```
### Functions
```
utils.prepare_batch: move to GPU and build target
ckpt.get_model_ckpt: load ckpt and substitue model weight and args
load_dataset.get_iterator: load data iterator {'train': , 'val': , 'test': }
```
## How To Use
### First check data loading
```
cd code
python3 cli.py check_dataloader
```

### Training
```
cd code
python3 cli.py train
```

### Evaluation
```
cd code
python3 cli.py evaluate --ckpt_name=$CKPT_NAME
```
- Substitute CKPT_NAME to your preferred checkpoint file, e.g., `ckpt_name=model_name_simclr_ckpt_3/loss_0.4818_epoch_15`
## Results

### References
- A lot of the codes are referenced from 
- https://github.com/PyTorchLightning/pytorch-lightning-bolts
- https://github.com/taoyang1122/pytorch-SimSiam
- https://github.com/zjcs/BYOL-PyTorch
- https://github.com/denn-s/SimCLR
- https://github.com/AidenDurrant/MoCo-Pytorch
and more. 

## Contact Me
To contact me, send an email to sally20921@snu.ac.kr
