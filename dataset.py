from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import scipy.io as sio
import os


# VideoDataset inherit from Dataset class
class VideoDataset(Dataset):
    def __init__(self, label_file, data_root):
        # list_file: data list
        self.list_files = pd.read_csv(label_file, sep=' ')
        self.root_dir = data_root

    def __getitem__(self, index):
        # get the path of videos from the label file
        video = os.path.join(self.root_dir, self.list_files.iloc[index, 0])
        frames = sio.loadmat(video)['A']
        label = sio.loadmat(video)['Y']
        return frames, label

    def __len__(self):
        return len(self.list_files)


def video_train_loader(config):
    return DataLoader(dataset=VideoDataset(config.train_label, config.data_root), num_workers=config.num_workers,
                      batch_size=config.train_batch_size, shuffle=True)


def video_test_loader(config):
    return DataLoader(dataset=VideoDataset(config.test_label, config.data_root), num_workers=config.num_workers,
                      batch_size=config.train_batch_size, shuffle=True)
