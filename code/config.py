config = { 
    'model_name': 'sim_siam', 
    'log_path': 'data/log',
    'batch_sizes':  (16, 24, 12),
    'use_inputs':['images'],
    'stream_type': ['visual'], #
    'cache_image_vectors': True,
    'datasets': 'imagenet',
    'image_path': 'data/imagenet', #data/cifar10 #data/cifar100
    'val_type': 'all', #  'all' | 'ch_only'
    'max_epochs': 100,
    'num_workers': 0, 
    'image_dim': 512,  # hardcoded for ResNet18
    'n_dim': 300,  
    'layers': 3,
    'dropout': 0.5,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'loss_name': 'sim_siam_loss',
    'optimizer': 'adam',
    'metrics': [],
    'log_cmd': True,
    'ckpt_path': 'data/ckpt',
    'ckpt_name': None,
    'shuffle': (False, False, False)
}


debug_options = {
    # 'image_path': './data/images/samples',
}

log_keys = [
    'model_name',
    # 'feature_pooling_method', # useless
]
