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
