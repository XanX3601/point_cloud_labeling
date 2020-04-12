import argparse
import os

import numpy as np
import sklearn.neighbors
import torch
import tqdm

import utils

RESULTS_DIR = "results"

if __name__ == "__main__":
    # Parser
    # ---------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", help="Is using CUDA", action="store_true")
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

    parser.add_argument(
        "--net",
        help="Neural net path",
        type=str,
        default="networks/neural_net.pt",
    )

    parser.add_argument(
        "--data",
        help="Point cloud path",
        type=str,
        default="dataset/numpy_files/vkitti3d_dataset_v1.0/06/fused06.npy",
    )

    args = parser.parse_args()

    # Results dir
    # ---------------
    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    # Using CUDA if asked and available
    # ---------------
    use_cuda = args.cuda and torch.cuda_is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load model
    # ---------------
    model = torch.load(args.net).to(device)

    # Main loop
    # ---------------
    point_cloud = np.load(args.data)
    predicted_labels = np.full((point_cloud.shape[0]), 13)

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
    print(voxel_centers.shape)

    # Compute KDTree
    # ---------------

    kdtree = sklearn.neighbors.KDTree(
        point_cloud[:, 0:3], metric="chebyshev")

    # Main computation
    # ---------------

    neighbors_indexes = kdtree.query_radius(
        voxel_centers, (args.R / args.N / 2))
    print(neighbors_indexes.shape)
    print(neighbors_indexes[0].shape)
    voxel_values = np.array([len(neighbors)
                             for neighbors in neighbors_indexes])

    nb_voxel_in_voxel_grid = args.N ** 3

    progress_bar = tqdm.trange(
        voxel_centers.shape[0],
        desc="Voxel centers",
        leave=False
    )

    for voxel_center_index in progress_bar:
        neighbors_filter = neighbors_indexes[voxel_center_index]
        points_in_voxel = point_cloud[
            neighbors_filter[voxel_center_index]
        ]

        if points_in_voxel.shape[0] > 0:

            matrix = voxel_values[
                voxel_center_index - nb_voxel_in_voxel_grid // 2:
                voxel_center_index + nb_voxel_in_voxel_grid // 2
            ]

            matrix = np.reshape(matrix, (1, 1, args.N, args.N, args.N))
            label = model(torch.from_numpy(matrix).float())
            predicted_labels[neighbors_filter] = label.item()

    # Create ply
    # ---------------
    utils.write_ply(
        "{}/{}.ply".format(RESULTS_DIR, args.data.split("/")
                           [-1].split(".")[-2]),
        [
            point_cloud[:, 0],
            point_cloud[:, 1],
            point_cloud[:, 2],
            predicted_labels.astype(np.int32),
            point_cloud[:, 6].astype(np.int32)
        ],
        ["x", "y", "z", "predicted_labels", "true_labels"],
    )
