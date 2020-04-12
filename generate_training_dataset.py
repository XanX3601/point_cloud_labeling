import argparse
import glob
import os
import random
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

    files = glob.glob(
        "{}/**/fused*.npy".format(utils.NPY_DIR), recursive=True)
    progress_bar = tqdm.tqdm(
        files,
        desc="Cloud points",
        unit="cp"
    )

    # Main loop
    # ---------------

    # For each point cloud
    for file_path in progress_bar:
        # We will use this point cloud for testing
        if file_path == "dataset/numpy_files/vkitti3d_dataset_v1.0/06/fused06.npy":  # noqa
            continue

        dataset_x = []
        dataset_y = []
        dataset_number_sample_per_label = {
            label: 0 for label in utils.LABELS.keys()}

        point_cloud = np.load(file_path)

        # Find label count and nb example per category
        # ---------------

        (values, counts) = np.unique(point_cloud[:, -1], return_counts=True)
        label_to_number_of_point = {
            values[i]: counts[i] for i in range(values.shape[0])
        }
        number_of_sample_per_label = np.min(counts)
        probability_per_label = {
            label: number_of_sample_per_label / label_to_number_of_point[label] for label in values  # noqa
        }

        # Compute bounb box
        # ---------------

        bound_box = utils.bound_box(point_cloud[:, 0:3])

        # Extend bound box to ensure that each voxel has enough neighbors
        bound_box_extended = list(bound_box)
        bound_box_extended[0] -= args.R / 2
        bound_box_extended[1] += args.R / 2
        bound_box_extended[2] -= args.R / 2
        bound_box_extended[3] += args.R / 2
        bound_box_extended[4] -= args.R / 2
        bound_box_extended[5] += args.R / 2

        # Compute grid
        # ---------------

        voxel_centers = utils.grid_centers(
            bound_box_extended,
            args.R / args.N,
        )

        # Compute KDTree
        # ---------------

        kdtree = sklearn.neighbors.KDTree(
            point_cloud[:, 0:3], metric="chebyshev")

        # Main computation
        # ---------------

        neighbors_indexes = kdtree.query_radius(
            voxel_centers, (args.R / args.N / 2))
        voxel_values = np.array([
            1 if len(neighbors) > 0 else 0 for neighbors in neighbors_indexes
        ])

        nb_voxel_in_voxel_grid = args.N ** 3

        sub_progress_bar = tqdm.trange(
            voxel_centers.shape[0],
            desc="Voxel centers",
            leave=False)
        for voxel_center_index in sub_progress_bar:
            points_in_voxel = point_cloud[
                neighbors_indexes[voxel_center_index]
            ]

            if points_in_voxel.shape[0] > 0:
                labels = points_in_voxel[:, -1]

                if (labels.shape[0] == 0):
                    voxel_label = 13
                else:
                    (values, counts) = np.unique(labels, return_counts=True)
                    label_index = np.argmax(counts)
                    voxel_label = values[label_index]

                if random.random() <= probability_per_label[voxel_label]:
                    matrix = voxel_values[
                        voxel_center_index - nb_voxel_in_voxel_grid // 2:
                        voxel_center_index + nb_voxel_in_voxel_grid // 2
                    ]

                    matrix = np.reshape(matrix, (1, args.N, args.N, args.N))

                    dataset_x.append(matrix)
                    dataset_y.append(voxel_label)
                    dataset_number_sample_per_label[voxel_label] += 1

        dataset_x = np.array(dataset_x)
        dataset_y = np.array(dataset_y)
        dataset_name = file_path.split("fused")[-1][:-4]

        with open("{}/{}.info".format(utils.TRAINING_DATASET_DIR, dataset_name), "w") as file_info:  # noqa
            file_info.write("Training dataset {}\n\n".format(dataset_name))

            file_info.write("number of sample: {}\n".format(
                dataset_x.shape[0])
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
                                     dataset_name), dataset_y)
