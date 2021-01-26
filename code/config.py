stages = ['pretrain', 'eval_linear', 'eval_semi', 'eval_transfer', 'eval_self']
models = ['moco', 'simclr', 'byol', 'swav', 'sim_siam']

log_keys = ['stage', 'model']
# naming convention: without s', short as possible, 'underscore'
# include _dataset, _transform in the name
# image path data/imagenet/train, data/imagenet/val
config = {
        # basic configuration
        'stage': 'pretrain',
        'model': 'sim_siam',
        'log_path': 'data/log',
        'ckpt_path': 'data/ckpt_path',
        'ckpt_name': None,
        'batch_size': 4096,
        'shuffle': False, 
        'epoch': 100,
        'worker': 0,
        'log_cmd': True,
        # distribution
        'distributed': False,
        'fp16_opt_level': '02',
        'sync_bn': True,
        # data loading
        'cache_image_vectors': False, 
        'dataset': 'imagenet', # 'mnist', 'stl10', 'cifar10', 'cifar100'
        'image_path': 'data/imagenet',
        'image_size': 224,
        'normalize': [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]],
        # cifar norm = [[0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]]
        'transform': 'sim_siam_transform',
        # model
        'use_inputs': ['x_i', 'x_j'],
        'use_outputs': ['p_i', 'p_j', 'z_i', 'z_j'], # simclr [p, z]
        'resnet': 'resnet50',
        'pretrained': False,
        'base_momentum': 0.996,
        'projection': 'mlp',
        'prj_dim': (2048, 256, 4096), # in_dim, out_dim, hsz
        'prediction': 'mlp',
        'prd_dim': (256, 256, 4096),
        # loss
        'loss': 'sim_siam_loss',
        'temperature': 0.1,
        # optimizer
        'optimizer': 'larc', # lars
        'sub_optimizer': 'sgd', # 'adam', 'adagrad', optimizer to wrap larc
        'lr': 1e-4, # 0.3 x batchsize / 256
        'lr_decay': 0,
        'weight_decay': 1e-6,
        'momentum': 0.9,
        # scheduler
        'scheduler': 'simclr_lr',
        'warm_up': 10,
        'step_size': 0, # for step lr decay
        'gamma': 0.1, # for step lr decay
        'cycle': 0.5, # for cosine lr decay
        'trust_coefficient': 1e-3, # for calculating lr
        'clip': True, # clipping/scaling mode of larc
        'eps': 1e-8, # caculating adaptive_lr
        # metric
        'metrics': [],
}

eval_linear = {
        # basic configuration
        'stage': 'eval_linear',
        'model': 'sim_siam',
        'log_path': 'data/log',
        'ckpt_path': 'data/ckpt_path',
        'ckpt_name': None,
        'batch_size': 4096,
        'shuffle': False, 
        'epoch': 100,
        'worker': 0,
        'log_cmd': True,
        # distribution
        'distributed': False,
        'fp16_opt_level': '02',
        'sync_bn': True,
        # data loading
        'cache_image_vectors': False, 
        'dataset': 'imagenet', # 'mnist', 'stl10', 'cifar10', 'cifar100'
        'image_path': 'data/imagenet',
        'image_size': 224,
        'normalize': [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]],
        # cifar norm = [[0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]]
        'transform': 'sim_siam_transform',
        # model
        'use_inputs': ['x_i', 'x_j'],
        'use_outputs': ['p_i', 'p_j', 'z_i', 'z_j'], # simclr [p, z]
        'resnet': 'resnet50',
        'pretrained': False,
        'base_momentum': 0.996,
        'projection': 'mlp',
        'prj_dim': (2048, 256, 4096), # in_dim, out_dim, hsz
        'prediction': 'mlp',
        'prd_dim': (256, 256, 4096),
        'num_classes': 100,
        # loss
        'loss': 'sim_siam_loss',
        'temperature': 0.1,
        # optimizer
        'optimizer': 'larc', # lars
        'sub_optimizer': 'sgd', # 'adam', 'adagrad', optimizer to wrap larc
        'lr': 1e-4, # 0.3 x batchsize / 256
        'lr_decay': 0,
        'weight_decay': 1e-6,
        'momentum': 0.9,
        # scheduler
        'scheduler': 'simclr_lr',
        'warm_up': 10,
        'step_size': 0, # for step lr decay
        'gamma': 0.1, # for step lr decay
        'cycle': 0.5, # for cos,ine lr decay
        'trust_coefficient': 1e-3, # for calculating lr
        'clip': True, # clipping/scaling mode of larc
        'eps': 1e-8, # caculating adaptive_lr
        # metric
        'metrics': [],
}

eval_semi = {
}

eval_transfer = {
}
