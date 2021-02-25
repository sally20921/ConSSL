# ConSSL: Contrastive Self-Supervised Learning SOTA Models
<p align="center">
  <img src="https://github.com/sally20921/ConSSL/blob/main/logo_size.jpg" />
	
<a href="https://sites.google.com/snu.ac.kr/serileeproject00/projects/"> Website </a> 

<a href="https://github.com/sally20921/ConSSL/blob/main/doc/doc.md/"> Documentation </a>

![PyPI Status](https://img.shields.io/badge/Package-PyPI-blue)
![MIT License](https://img.shields.io/github/license/sally20921/ConSSL)
![Website](https://img.shields.io/badge/website-up-yellowgreen)
</p>

 


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

# train from scratch
model = SimSiam()
trainer = Trainer(gpu=4)
trainer.fit(
	model,
	DataLoader(train_dataset),
	DataLoader(val_dataset),
)
```

## Notes On Implementation

- I found that using SimCLR augmentation directly will sometimes cause the model to collpase. This maybe due to the fact that SimCLR augmentation is too strong.
- Adopting the MoCo augmentation during the warmup stage helps.
- Gradient check for Batch-Optimization: Gradient descent over a batch of samples can not only benefit the optimization but also leverages data parallelism. However, you have to be careful not to mix data across the batch dimension. Only a small error in a reshape or permutation operation results in the optimization getting stuck and you won't even get a runtime eror. You should check the operations that reshape and permute tensor dimensions in your model. 
* run the model on an example batch (can be random data)
* get the output batch and select the n-th sample (choose n)
* compute a dummy loss value of only that sample and compute the gradient w.r.t. the entire input batch
* observce that only the i-th sample in the input batch has non-zero gradient
```python
from pytorch_lightning import Trainer
from ConSSL.callbacks import BatchGradientVerificationCallback

model = YourLightningModule()
verification = BatchGradientVerificationCallBack()
trainer = Trainer(callbacks=[verification])
trainer.fit(model)
````
- this is how you should predict based on ConSSL models in your own data
```python
# trained without labels 
from ConSSL.models.self_supervised import SimCLR
weight_path = 'path/to/your/checkpoint/file'
simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
resnet50_unsupervsied = simclr.encoder.eval()

# trained with labels 
from torchvision.models import resnet50
resnet50_supervised = resnet50(pretrained=True)

x = image_sample()
unsup_feats = resnet50_unsupervsied(x)
sup_feats = resnet50_supervised(x)
```

## Dataset

Collection of useful datasets including STL10, MNIST, CIFAR10, CIFAR100, ImageNet.
```python
from ConSSl.datamodules import CIFAR10DataModule, ImagenetDataModule, MNISTDataModule, STL10DataModule
# ImagenetDataModule assumes you have ILSVRC2012 imagenet data downloadedd. It validates the data using meta.bin.

# datamodules for debugging
from ConSSL.datasets import DummyDataset
from torch.utils.data import DataLoader 

# mnist dims
ds = DummyDataset((1,28,28), (1,))
dl = DataLoader(ds, batch_size=256)
# get first batch
batch = next(iter(dl))
x, y = batch
x.size() # torch.Size([256, 1, 28, 28])
y.size() # torch.Size([256,1])

# standard transforms is defined as follows:
mnist_transforms = transform_lib.Compose([transform_lib.ToTensor()])

dm = CIFAR10DataModule(PATH)
dm.train_transforms = 
dm.test_transforms = 
dm.val_transforms = 
model = LiteModel()

Trainer().fit(model, datamodule=dm)
```



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
#### SimCLR
##### Usage 
```python
import pytorch_lightning as pl
from ConSSL.models.self_supervised import SimCLR
from ConSSL.datamodules import CIFAR10DatatModule
from ConSSL.models.self_supervised.simclr.transforms import (SimCLREvalDataTransform, SimCLRTrainDataTransform)

# data
dm = CIFAR10DataModule(num_workers=0)
dm.train_transforms = SimCLRTrainDataTransform(32)
dm.val_transforms = SimCLREvalDataTransform(32)

# model
model = SimCLR(num_samples=dm.num_samples, batch_size=dm.batch_size, dataset='cifar10')

# fit 
trainer = pl.Trainer()
trainer.fit(model, datamodule=dm)

#to finetune
python simclr_finetuner.py --gpus 4 --ckpt_path path/to/simclr/ckpt --dataset cifar10 --batch_size 64 --num_workers 8 --learning_rate 0.3 --num_epochs 100

```
![simclr_pretrain](https://github.com/sally20921/ConSSL/blob/main/doc/simclr_pretraining.png)


##### Results
|Implementation| Dataset     | Architecture | Optimizer|Batch size | Epochs | Linear Evaluation| 
|--------------| ------------| ------------ | ---------|-----------| ------ | -----------------|
|   Original   | CIFAR10     | ResNet50     | LARS     |512       | 500    | 0.72             | 
|   Mine       | CIFAR10     | ResNet50     | LARS-SGD |512       | 200    | 0.68             | 
|   Original   | imagenet     | ResNet50     | LARS     |256       | 500   | 0.67             | 
|   Mine       | imagenet    | ResNet50     | LARS-SGD |256      | 200    | 0.63            | 

#### to reproduce
```
cd code
# change the configuration setting in config.py
python cli.py pretrain
python cli.py linear_evaluation
```

#### MoCov2
```python
# CLI command
# imagenet
python moco2_module.py --gpus 8 --dataset imagenet2012 --data_dir /path/to/imagenet --meta_dir /path/to/folder/with/meta.bin/ --batch_size 512

ConSSL.models.self_supervised.MocoV2(base_encoder='resnet18', emb_dim=128, num_negatives=65536, encoder_momentum=0.999, softmax_temperature=0.07, learning_rate=0.03. momentum=0.9, weight_decay=0.0001, data_dir='./', batch_size=256, use_mlp=False, num_workers=8, *args, **kwargs)

'''Args:
base_encoder: torchvision model name or toch.nn.Module
emb_dim: feature dimension
num_negatives: queue size
encoder_momentum: moco momentum of updating key encoder
use_mlp: add an mlp to the encoder
'''
from ConSSL.models.self_supervised import MocoV2 
model = MocoV2()
trainer = Trainer()
trainer.fit(model)
```
##### Results
|Implementation| Dataset     | Architecture | LR       |Batch size | Epochs | Linear Evaluation| 
|--------------| ------------| ------------ | ------- |-----------| ------ | -----------------|
|  Mine  | ImageNet    | ResNet50     | Cosine |512      | 200   |      0.65  | 

#### BYOL
```python
from ConSSL.callbacks.byol_updates import BYOLMAWeightUpdate

'''the exponential moving average weight update rule from BYOL.
Your model should have self.online_network, self.target_network'''

model = Model()
model.online_network = 
model.target_network = 

trainer = Trainer(callbacks=[BYOLMAWeightUpdate(initial_tau=0.996)])
```

```python
# CLI command
python byol_module.py --gpus 8 --dataset imagenet2012 --data_dir /path/to/imagenet/ --meta_dir /path/to/folder/with/meta.bin/ --batch_size 512

ConSSL.models.self_supervised.BYOL(num_classes, learning_rate=0.2, weight_decay=1.5e-6, input_height=32,
batch_size=32, num_workers=0, warmup_epochs=10, max_epochs=1000, **kwargs)
```

##### Results
|Implementation| Dataset     | Architecture | LR       |Batch size | Epochs | Linear Evaluation| 
|--------------| ------------| ------------ | ------- |-----------| ------ | -----------------|
|   Original   | ImageNet    | ResNet50     | Cosine |4096      | 300   | 0.72          | 
|   Mine       | ImageNet    | ResNet50     | Cosine |512       | 200   | 0.66             | 
#### SwAV
```python
import pytorch_lightning as pl
from ConSSL.models.self_supervised import SwAV
from ConSSL.datamodules import STL10DataModule
from ConSSL.models.self_supervised.swav.transform import (SwAVTrainDataTransform, SwAVEvalDataTransform)
from ConSSL.transforms.dataset_normalization import stl10_normalization

# data 
batch_size = 128
dm = STL10DataModule(data_dir='.', batch_size=batch_size)
dm.train_dataloader = dm.train_dataloader_mixed
dm.val_dataloader = dm.val_dataloader_mixed

dm.train_transforms = SwAVTrainDataTransform(normalize=stl10_normalization())
dm.val_transforms = SwAVEvalDataTransform(normalize=stl10_normalization())

# model
model = SwAV(gpus=1, num_samples=dm.num_unlabeled_samples, dataset='stl10', batch_size=batch_size)

# fit 
trainer = pl.Trainer(precision=16)
trainer.fit(model)
```
![swav_pretrain](https://github.com/sally20921/ConSSL/blob/main/doc/swav_lr.png)
![swav_lr](https://github.com/sally20921/ConSSL/blob/main/doc/swav_pretrain.png)
##### Results
|Implementation| Dataset     | Architecture | Optimizer|Batch size | Epochs | Linear Evaluation| 
|--------------| ------------| ------------ | ---------|-----------| ------ | -----------------|
|   Mine       | STL10       | ResNet50     | LARS-SGD |128        | 100    | 0.86             | 

#### to reproduce
```
cd code
# change the configuration setting in config.py
python cli.py pretrain
python cli.py linear_evaluation
```
#### SimSiam
|Implementation| Dataset     | Architecture         |Batch size | Epochs | Linear Evaluation| 
|--------------| ------------| ------------  |-----------| ------ | -----------------|
|   Original   | CIFAR10    | ResNet18      |512      | 800   | 0.91          | 
|   Mine       | CIFAR10   | ResNet18     |512       | 300   | 0.72             | 

### Linear Evaluation Protocol
```python
from pytorch_lightning as pl
from ConSSL.models.regression import LogisticRegression
from ConSSL.datamodules import ImagenetDataModule

imagenet = ImagenetDataModule(PATH)

# 224x224x3
pixels_per_image = 150528
model = LogisticRegression(input_dim=pixels_per_image, num_classes=1000)
model.prepare_data = imagenet.prepare_data
trainer = Trainer(gpus=2)
trainer.fit(model, imagenet.train_dataloader(batch_size=256), imagenet.val_dataloader(batch_size=256))
````

### Semi-Supervised Learning
use imagenet subset from https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/image_classification
1. Unfrozen Finetuning
```python
from ConSSL.models.self_supervised import SimCLR
from ConSSL.models.regression import LogisticRregression

weight_path = 'checkpoint/path'
simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
resnet50 = simclr.encoder 
# don't call simclr.freeze()

classifier = LogisticRegresion()
for (x,y) in own_data:
 feats = resnet50(x)
 y_hat = classifier(feats)
```
2. Freeze then Unfreeze
```python
from ConSSL.models.self_supervised import SimCLR
from ConSSL.models.regression import LogisticRregression

weight_path = 'checkpoint/path'
simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
resnet50 = simclr.encoder 
resnet50.eval()

classifier = LogisticRegression()
for epoch in epochs:
 for (x,y) in own_data:
  feats = resnet50(x)
  y_hat = classifier(feats)
  loss = cross_entropy_with_logits(y_hat, y)
 
 # unfreeze after 10 epochs 
 if epoch == 10:
  resnet.unfreeze()
```

### Transfer Learning
```python
from pytorch_lightning as pl
from ConSSL.models.regression import LogisticRregression
from ConSSL.datamodules import MNISTDataModule

dm = MNISTDataModule(num_workers=0, data_dir=tmpdir)

model= LogisticRegression(input_dim=28*28, num_classes=10, learning_rate=0.001)
model.prepare_data = dm.prepare_data
model.train_dataloader = dm.train_dataloader
model.val_dataloader = dm.val_dataloader
model.test_dataloader = dm.test_dataloader

trainer = pl.Trainer(max_epochs=200)
trainer.fit(model)
trainer.test(model)
```
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

```python
from ConSSL.callbacks.ssl_online import SSLOnlineEvaluator

''' attaches a MLP for fine-tuning using the standard self-supervised protocol'''
model = Model()
model.z_dim = # the representation dim
model.num_classes = # the number of classes in the model

online_eval = SSLOnlineEvaluator(z_dim=model.z_dim, num_classes=model.num_classes, dataset='imagenet')
# if the dataset if stl10, you need to get the labeled batch
```
## Callback

A callback is a self-contained program that can be intertwined into a training pipeline. 

```python
from ConSSL.callbacks import import Callback

class MyCallback(Callback):
 def on_epoch_end(self, trainer, pl_module):
  # do something
```

The data monitoring callbacks allow you to log and inspect the distribution of data that passes through 
the training step and layers of the model. 

```python
from ConSSL.callbacks import TrainingDataMonitor
from pytorch_lightning import Trainer

monitor = TrainingDataMonitor(log_every_n_steps=25)

model = YourLightningModule()
trainer = Trainer(callbacks=[monitor])
trainer.fit()
```


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
