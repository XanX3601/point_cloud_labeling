import glob
import os

import numpy as np
import tqdm

import utils

NPY_DIR = "dataset/numpy_files"
PLY_DIR = "dataset/ply_files"

if __name__ == "__main__":
    if not os.path.exists(PLY_DIR):
        os.mkdir(PLY_DIR)

    # Get all npy files paths
    files = glob.glob("{}/**/*.npy".format(NPY_DIR), recursive=True)

    # For each file, generate its ply file
    for f in tqdm.tqdm(files):
        data = np.load(f)

        # File name without extension
        file_name = f.split("/")[-1][:-4]

        utils.write_ply(
            "{}/{}.ply".format(PLY_DIR, file_name),
            [data[:, 0], data[:, 1], data[:, 2],
                data[:, 3].astype(np.uint8), data[:, 4].astype(np.uint8),
                data[:, 5].astype(np.uint8), data[:, 6].astype(np.int32)],
            ["x", "y", "z", "red", "green", "blue", "label"],
        )
