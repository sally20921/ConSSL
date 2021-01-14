# Short Explanation on PyTorch Ignite
## Engine
Engine is an abstraction that loops a given number of times over provided data,
executes a processing function and returns a result:
```python
while epoch < max_epochs:
 # run an epoch on data
 data_iter = iter(data)
 while True:
  try: 
   batch = next(data_iter)
   output = process_function(batch)
   iter_counter += 1
  except StopIteration:
   data_iter = iter(data)

 if iter_counter == epoch_length:
  break
```

```python
def train_step(trainer, batch):
 data_1 = batch["data_1"]
 data_2 = batch["data_2"]
 # ...

 model_1.train()
 optimizer_1.zero_grad()
 loss_1 = forward_pass(data_1, model_1, criterion_1)
 loss_1.backward()
 optimizer_1.step()
 # ...

 model_2.train()
 optimizer_2.zero_grad()
 loss_2 = forward_pass(data_2, model_2, criterion_2)
 loss_2.backward()
 optimizer_2.step()
 # ...

 # User can return any type of structure
 return {
  "loss_1": loss_1,
  "loss_2": loss_2,
  # ...
 }

trainer = Engine(train_step)
trainer.run(data, max_epochs=100)
```
batch in `train_step` function is user-defined and can contain any data required for single iteration. 


