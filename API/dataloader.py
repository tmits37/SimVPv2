from .dataloader_moving_mnist import load_data as load_mmnist
from .dataloader_s4a import load_data as load_s4a


def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, **kwargs):
    if dataname == 'mmnist':
        return load_mmnist(batch_size, val_batch_size, num_workers, data_root)
    elif dataname == 's4a':
        return load_s4a(batch_size, val_batch_size, num_workers, data_root)