from torch import optim

class SGD(optim.SGD):
    @classmethod
    def resolve_args(cls, args, params):
        options = {}
        options['lr'] = args.get("learning_rate", 0.01)
        #options['lr_decay'] = args.get("lr_decay", 0)
        options['weight_decay'] = args.get("weight_decay", 0)
        options['momentum'] = args.get("momentum", 0.9)
        return cls(params, **options)

