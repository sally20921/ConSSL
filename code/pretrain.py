'''
get_transform(args, eval_stage)
get_dataset(args, transform, eval_stage)
In pretraining stage, eval_stage set to 'none'
'''
from dataloader import get_transform
from dataloader import get_dataset

from utils import prepare_batch
from logger import get_logger, log_results, log_results_cmd

from ignite.engine.engine import Engine, State, Events
from ignite.metric import Loss

import numpy as np
from apex import amp
import ignite.distributed as idist
from ignite.contrib.engines import common

def get_trainer(args, model, loss_fn, optimizer, scheduler):
    def update_model(trainer, batch):
        model.train()
        optimizer.zero_grad()

        # to gpu
        net_inputs, target = prepare_batch(args, batch)
        # ** : dictionary input to each argument
        # y_pred : dict {z_i, z_j, p_i, p_j}
        y_pred = model(**net_inputs)
        batch_size = target.shape[0] # N
        loss = loss_fn(y_pred)
        loss = loss.mean() # ddp

        #with amp.scale_loss(loss, optimizer, loss_id=0) as scaled_loss:
        #    scaled_loss.backward()
        loss.backward()
        optimizer.step()
        scheduler.step()

        return loss.item(), batch_size, y_pred.detach()

    trainer = Engine(update_model)

    metrics = {
            'loss': Loss(output_transform=lambda x:(x[0], x[1])),
            }

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer


def pretrain(args):
    tf = get_transform(args, 'none')
    ds = get_dataset(args, tf, 'none')

    args, model, ckpt_available = get_model_ckpt(args)

    if ckpt_available:
        print("loaded checkpoint {} in pretraining stage".format(args.ckpt_name))
        loss_fn = get_loss(args)
        sub_optimizer = get_sub_optimizer(args, model)
        optimizer = get_optimizer(args, sub_optimizer)
        scheduler = get_scheduler(args, optimizer)

        # setup nvidia/apex amp
        # model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level, num_losses=1)
        # model = idist.auto_model(model)

        trainer = get_trainer(args, model, loss_fn, optimizer, scheduler)

        metrics = get_metrics(args)
        logger = get_logger(args)

        @trainer.on(Events.STARTED)
        def on_training_started(engine):
            print("Begin Pretraining")

        # batch-wise
        @trainer.on(Events.ITERATION_COMPLETED)
        def log_iter_results(engine):
            log_results(logger, 'pretrain/iter', engine.state, engine.state.iteration)

        # epoch-wise (ckpt)
        @trainer.on(Events.EPOCH_COMPLETED)
        def save_epoch(engine):
            log_results(logger, 'pretrain/epoch', engine.state, engine.state.epoch)
            log_results_cmd(logger, 'pretrain/epoch', engine.state, engine.state.epoch)
            save_ckpt(args, engine.state.epoch, engine.state.metrics['loss'], model)

        trainer.run(ds, max_epochs=args.epoch)




