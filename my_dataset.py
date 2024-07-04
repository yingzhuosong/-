from torch.utils.data import Dataset
import os
from scipy.io import loadmat
import torch


class MyDataset(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.signal_path = os.path.join(self.root_dir, self.label_dir)
        self.signal_name = os.listdir(self.signal_path)

    def __getitem__(self, idx):
        signal_name = self.signal_name[idx]
        signal_item_path = os.path.join(self.root_dir, self.label_dir, signal_name)
        signal = loadmat(signal_item_path)
        if self.label_dir == "9" or self.label_dir=="4":
            signal = torch.tensor(signal["Newdata"], dtype=torch.float)
        else:
            signal = torch.tensor(signal["ECG"][0][0][2], dtype=torch.float)
        padding_length = 1000
        if signal.shape[1] < padding_length:
            pad_tensor = torch.zeros(12, padding_length-signal.shape[1], dtype=torch.float)
            signal = torch.cat([signal, pad_tensor], dim=1)
        else:
            signal = signal[:, 0:padding_length]
        label = int(self.label_dir)-1
        return signal, label

    def __len__(self):
        return len(self.signal_name)


def combine(root_dir):

    label_dir_1 = "1"
    label_dir_2 = "2"
    label_dir_3 = "3"
    label_dir_4 = "4"
    label_dir_5 = "5"
    label_dir_6 = "6"
    label_dir_7 = "7"
    label_dir_8 = "8"
    label_dir_9 = "9"
    my_dataset_1 = MyDataset(root_dir, label_dir_1)
    my_dataset_2 = MyDataset(root_dir, label_dir_2)
    my_dataset_3 = MyDataset(root_dir, label_dir_3)
    my_dataset_4 = MyDataset(root_dir, label_dir_4)
    my_dataset_5 = MyDataset(root_dir, label_dir_5)
    my_dataset_6 = MyDataset(root_dir, label_dir_6)
    my_dataset_7 = MyDataset(root_dir, label_dir_7)
    my_dataset_8 = MyDataset(root_dir, label_dir_8)
    my_dataset_9 = MyDataset(root_dir, label_dir_9)
    my_dataset = (my_dataset_1 + my_dataset_2 + my_dataset_3 + my_dataset_4 + my_dataset_5
                      + my_dataset_6 + my_dataset_7 + my_dataset_8 + my_dataset_9)

    return my_dataset


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    my_dataset = combine(root_dir="training_set")
    signal, label = my_dataset[706]
    plt.plot(signal[11, :])
    plt.show()
    print(signal.shape)
    print(len(my_dataset))





