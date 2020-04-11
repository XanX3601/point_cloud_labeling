import torch
import torch.utils.data as data


class Dataset(data.dataset.Dataset):
    def __init__(self):
        self.x = torch.randint(
            0, 2, (320, 1, 20, 20, 20), dtype=torch.float32)
        self.y = torch.randint(0, 14, (320,))

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
