# Classic ML Models
This module implements classic machine learning models in PyTorch Lightning, including linear regression and logsitic regression.
Here I use PyTorch to enable multi-GPU and half-precision training.

## Linear Regression
* Linear regression fits a linear model between a real-valued target variable *y* and one or more features *X*. 
* Estimate the regression coefficients that minimize the mean squared error between the predicted and true target values.

* I formulated the linear regression model as a single-layer neural network. 
* By default, I include only one neuron on the output layer, although you can specify the *output_dim* yourself.

* Add either L1 or L2 regularization, or both, by specifying the regularization strength (default 0).
```python
from ConSSL.models.regression import LinearRegression
import pytorch_lightning as pl
from ConSSL.datamodules import SKlearnDataModules
from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)
loaders = SKlearnDataModule(X, y)

model = LinearRegression(input_dim=13)
trainer = pl.Trainer()
trainer.fit(model, trian_dataloader=loaders.train_dataloader(), val_dataloaders=loaders.val_dataloader())
trainer.test(test_dataloaders=loaders.test_dataloader())
```

## Logistic Regression
* Logistic regression is a linear model used for classification, i.e. when we have a categorical target variable. 
This implementation supports both binary and multi-class classification.

```python
from sklearn.datasets import load_iris
from ConSSL.models.regression import LogisticRegression
from ConSSL.datamodules import SKlearnDataModule, MNISTDataModule
import pytorch_lightning as pl

dm = MNISTDataModule(num_workers=0, data_dir=tmpdir)

model = LogisticRegression(input_dim=28*28, num_classes=10, learning_rate=0.001)
model.prepare_data = dm.prepare_data
model.train_dataloader = dm.train_dataloader
model.val_dataloader = dm.val_dataloader
model.test_dataloader = dm.test_dataloader

trainer = pl.Trainer(max_epochs=200)
trainer.fit(model)
trainer.test(model)
# {test acc: 0.92}
```

# Self-Supervised Learning
## Extracting Image Features 
* The models in this module are trained unsupervised and thus can capture better image representations or features.
```python
from ConSSL.models.self_supervised import SimCLR

weight_path = 'simclr/imagenet/weights/checkpoint/file'
simclr = SimCLR.load_from_checkpoint(weight_path, strict=False0

simclr_resnet50 = simclr.encoder
simclr_resnet50.eval()

my_dataset = SomeDataset()
for batch in my_dataset:
 x, y = batch
 out = simclr_resnet50(x)
```
* This means you can now extract image representations that were pretrained via unsupervised learning.

# Learning Rate Scheduler
## Linear Warmup Cosine Annealing Learning Rate Scheduler
```python
from ConSSL.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

layer = nn.Linear(10,1)
optimizer = Adam(layer.parameters(), lr=0.02)
scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epoch=10, max_epochs=100)

for epoch in range(100):
 # ...
 scheduler.step()
 ```
 
 # Self-Supervised Learning Transforms
 ## SimCLR Transforms
 ```from ConSSL.models.self_supervised.simclr.transforms import SimCLRDTrainDataTransforms, SimCLREvalDataTransforms
 
 train_transforms = SimCLRTrainDataTransform(input_height=32)
 eval_transforms = SimCLREvalDataTransform(input_height=32)
 x = sample()
 (xi, xj) = train_transform(x)
 (xi, xj) = eval_transform(x)
```

# Utils
## Identity class 
```python
from ConSSL.utils import Identity
model = resnet18()
model.fc = Identity()
```
## SSL-ready resnets
* torchvision resnets with the fc layers removed and with the ability to return feature maps instead of just the last one 
```python
from ConSSL.utils.self_supervised import torchvision_ssl_encoder

resnet = torchvision_ssl_encoder('resnet18', pretrained=False, return_all_features_maps=True)
x = torch.rand(3,3,32,32)
feat_maps = resnet(x)
```

## SSL backbone finetuner
* finetunes a self-supervised learning backbone using the standard evaluation protocol of a single layer MLP with 1024 units
```python 
from ConSSL.self_supervised import SSLFineTuner
from ConSSL.models.self_supervised import SimCLR
from ConSSL.datamodules import CIFAR10DataModule
from ConSSl.models.self_supervised.simclr.transforms import SimCLREvalTransforms, SimCLRTrainTransforms

# pretrained model
backbone = SimCLR.load_from_checkpoint(PATH, strict=False)

# dataset + transforms
dm = CIFAR10DataModule(data_dir='.')
dm.train_transforms = SimCLRTrainTransforms
dm_val_transforms = SimCLREvalTransforms

# finetuner
finetuner = SSLFineTuner(backbone, in_features=backbone.z_dim, num_classes=backbone.num_classes)

# train
trainer = pl.Trainer()
trainer.fit(finetuner, dm)

# test
trainer.test(datamodule=dm)
```
# Semi-Supervised Learning
* collection of utilities for semi-supervised learning where some part of the data is labeled and the other part is not.
## Balanced Classes
* Makes sure each batch has an equal amount of data from each class.
```python
from ConSSL.utils.semi_supervised import balance_classes

'''args:
X : input features
Y: mixed labels (ints)
batch_size: the ultimate batch size
'''
```

## Half-Labeled Batches
* given a labeled dataset and an unlabeled dataset, the function generates a joint pair where half the batches are labeled and the other half is not.
```python
from ConSSL.utils.semi_supervised import balance_classes

'''
ConSSL.utils.semi_supervised.generate_half_labeled_batches(smaller_set_X, smaller_set_Y, larger_set_X, larger_set_Y, batch_size)
'''
```
