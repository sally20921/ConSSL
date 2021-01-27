import logging 

from tqdm import tqdm
from tensorboardX import SummaryWriter
from tensorboard import default, program
import tensorflow as tf

from utils import get_dirname_from_args, get_now

class Logger:
    def __init__(self, args):
        # also show log in commandline?
        self.log_cmd = args.log_cmd

        log_name = get_dirname_from_args(args)
        log_name += '_{}'.format(get_now())

        self.log_path = args.log_path / log_name
        self.log_path.mkdir(parents=True, exist_ok=True)

        # also show tensorboard
        self.tfboard = SummaryWriter(self.log_path)
        self.url = run_tensorboard(self.log_path)
        print("Running Tensorboard at {}".format(self.url))

    def __call__(self, name, val, n_iter):
        # save name, value, number of iteration
        self.tfboard.add_scalar(name, val, n_iter)
        if self.log_cmd:
            tqdm.write('{}:({},{})'.format(n_iter, name, val))

def run_tensorboard(log_path):
    log = logging.getLogger('tensorflow').setLevel(logging.ERROR)

    port_num = abs(hash(log_path))
    tb = program.TensorBoard(default.get_plugins(), get_assets_zip_provider())
    tb.configure(argv=[None, '--logdir', str(log_path), '--port', str(port_num), '--samples_per_plugin', 'text=100'])

    url = tb.launch()
    return url

# forward compatibility for version > 1.12
def get_assets_zip_provider():
    path = os.path.join(tf.resource_loader.get_data_files_path(), 'webfiles.zip')
    if not os.path.exists(path):
        print('webfiles.zip static assets not found: %s', path)
        return None
    return lambda: open(path, 'rb')

def log_results(logger, name, state, step):
    for key, val in state.metrics.items():
        if isinstance(val, dict):
            for key2, v in val.items():
                logger("{}/{}/{}".format(name, key, key2), v, step)
        else:
            logger("{}/{}".format(name, key), val, step)

def log_results_cmd(name, state, step):
    for key, val in state.metrics.items():
        if isinstance(val, dict):
            for key2, v in val.items():
                print("{}/{}/{}".format(name, key, key2), v, "step:{}".format(step))

        else:
            print("{}/{}".format(name, key), val, "step:{}".format(step))

def get_logger(args):
    return Logger(args)
