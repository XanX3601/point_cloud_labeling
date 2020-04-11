import os
import sys

import numpy as np
import tqdm

import utils

if __name__ == "__main__":
    if not os.path.exists(utils.NPY_DIR):
        print("Dataset cannot be found", file=sys.stderr)
        exit(1)

    progress_bar = tqdm.tqdm(utils.POINT_CLOUD_DIRS)

    for directory_path in progress_bar:
        point_clouds = []
        for file_ in os.listdir(directory_path):
            point_cloud = np.load("{}/{}".format(directory_path, file_))
            point_clouds.append(point_cloud)

        point_cloud = np.concatenate(point_clouds, axis=0)

        np.save(
            "{}/fused{}.npy".format(directory_path,
                                    directory_path.split('/')[-1]),
            point_cloud
        )
