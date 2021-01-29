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



