import numpy as np
import torch
import torch.utils.data as data

NPY_DIR = "dataset/numpy_files"

POINT_CLOUD_DIRS = [
    "{}/vkitti3d_dataset_v1.0/0{}".format(NPY_DIR, i) for i in range(1, 7)
]

TRAINING_DATASET_DIR = "dataset/training_dataset"


class Dataset(data.dataset.Dataset):
    def __init__(self):
        """For the moment we keep 06 for testing."""
        self.x = []
        self.y = []

        for i in range(1, 6):
            x = np.load("{}/0{}_x.npy".format(TRAINING_DATASET_DIR, i))
            y = np.load("{}/0{}_y.npy".format(TRAINING_DATASET_DIR, i))
            self.x.append(x)
            self.y.append(y)

        self.x = np.concatenate(self.x, axis=0)
        self.y = np.concatenate(self.y, axis=0)

        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).long()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


LABELS = {
    0: "terrain",
    1: "tree",
    2: "vegetation",
    3: "building",
    4: "road",
    5: "guard_rail",
    6: "traffic_sign",
    7: "traffic_light",
    8: "pole",
    9: "misc",
    10: "truc",
    11: "car",
    12: "van",
    13: "none",
}
