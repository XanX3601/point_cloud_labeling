import argparse
import glob
import os
import sys

import numpy as np
import sklearn.neighbors
import tqdm

import utils

if __name__ == "__main__":
    # Parser
    # ---------------

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-R",
        help="the size of the cube. 6 by default.",
        type=float,
        default=6
    )

    parser.add_argument(
        "-N",
        help="the number of cell in a cube. 20 by default.",
        type=int,
        default=20
    )

    args = parser.parse_args()

    # Check dataset avalaible
    # ---------------

    if not os.path.exists(utils.NPY_DIR):
        print("Dataset cannot be found", file=sys.stderr)
        exit(1)

    # Check if training dataset dir exists
    # ---------------
    if not os.path.exists(utils.TRAINING_DATASET_DIR):
        os.mkdir(utils.TRAINING_DATASET_DIR)

    # Retrieve file paths
    # ---------------

    files = glob.glob("{}/**/fused*.npy".format(utils.NPY_DIR), recursive=True)
    progress_bar = tqdm.tqdm(
        files,
        desc="Cloud points",
        unit="cp"
    )

    # Main loop
    # ---------------

    # For each point cloud
    for file_path in progress_bar:
        dataset_x = []
        dataset_labels = []
        dataset_number_sample_per_label = {
            label: 0 for label in utils.LABELS.keys()
        }

        point_cloud = np.load(file_path)

        voxel_grid_centers = utils.grid_centers(
            utils.bound_box(point_cloud[:, 0:3]),
            args.R
        )

        kdtree = sklearn.neighbors.KDTree(
            point_cloud[:, 0:3], metric="chebyshev")

        sub_progress_bar = tqdm.tqdm(
            voxel_grid_centers, leave=False, desc="Voxel grids", unit="vg")

        # For each voxel grid
        for center_voxel_grid in sub_progress_bar:
            # Compute voxel grid values
            # ---------------

            # Compute center of each voxel in the voxel grid
            voxel_centers = utils.grid_centers(
                [
                    center_voxel_grid[0] - args.R / 2,
                    center_voxel_grid[0] + args.R / 2,
                    center_voxel_grid[1] - args.R / 2,
                    center_voxel_grid[1] + args.R / 2,
                    center_voxel_grid[2] - args.R / 2,
                    center_voxel_grid[2] + args.R / 2,
                ],
                args.R / args.N
            )

            # Find the number oi points in each voxel
            number_of_point_per_voxel = kdtree.query_radius(
                voxel_centers, (args.R / args.N) / 2, count_only=True)
            # Reshape the result
            voxel_grid_values = np.reshape(number_of_point_per_voxel,
                                           (1, args.N, args.N, args.N))
            voxel_grid_values[voxel_grid_values > 0] = 1

            dataset_x.append(voxel_grid_values)

            # Compute voxel grid label
            # ---------------

            point_filter = kdtree.query_radius(np.reshape(
                center_voxel_grid, (1, -1)), args.R)[0]
            labels = point_cloud[point_filter][:, -1]

            if (labels.shape[0] == 0):
                dataset_labels.append(13)
                dataset_number_sample_per_label[13] += 1
            else:
                (values, counts) = np.unique(labels, return_counts=True)
                label_index = np.argmax(counts)
                label = values[label_index]
                dataset_number_sample_per_label[label] += 1
                dataset_labels.append(label)

        dataset_x = np.array(dataset_x, dtype=np.float32)
        dataset_labels = np.array(dataset_labels, dtype=np.float32)
        dataset_name = file_path.split("fused")[-1][:-4]

        with open("{}/{}.info".format(utils.TRAINING_DATASET_DIR, dataset_name), "w") as file_info:  # noqa
            file_info.write("Training dataset {}\n\n".format(dataset_name))

            file_info.write("number of sample: {}\n".format(
                voxel_grid_centers.shape[0])
            )

            file_info.write("Label info:\n"),
            for label, label_name in utils.LABELS.items():
                file_info.write(
                    "    number of {}-{}: {}\n".format(
                        label,
                        label_name,
                        dataset_number_sample_per_label[label]
                    )
                )
            file_info.write("R: {}\n".format(args.R))
            file_info.write("N: {}\n".format(args.N))

        np.save("{}/{}_x.npy".format(utils.TRAINING_DATASET_DIR,
                                     dataset_name), dataset_x)
        np.save("{}/{}_y.npy".format(utils.TRAINING_DATASET_DIR,
                                     dataset_name), dataset_labels)
