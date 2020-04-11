# point_cloud_labelling

In this project, we try to implement the method described in [this paper](https://ieeexplore.ieee.org/abstract/document/7900038). You can also find it in [`materials`](materials/). The goal is to labelize a point cloud without a previous "segmentation step or hand-crafted features". To do so, we will use a 3-dimensional convolutional artificial neural network.

## Prerequisites

You need to have Python 3.7 (at least) installed on your computer. To begin, execute the following commands.

```shell
git clone https://github.com/XanX3601/point_cloud_labeling.git
pip install -r requirements.txt
```

## Dataset

For this project, we use the [vkitti3D-dataset](https://github.com/VisualComputingInstitute/vkitti3D-dataset.git). You can download it with `wget` or `curl` and unzip it with `unzip` or another program.

```shell
curl -o dataset/dataset.zip https://www.vision.rwth-aachen.de/media/resource_files/vkitti3d_dataset_v1.0.zip
wget -O dataset/dataset.zip https://www.vision.rwth-aachen.de/media/resource_files/vkitti3d_dataset_v1.0.zip
unzip dataset/dataset.zip -d dataset/numpy_files
```

A script allows you to fuse each cloud points from the dataset into a unique
clous point. These fused cloud points are used to generate the training
datasets.

```shell
python fuse_npy.py
```

A script allows you to transform the fused cloud points into training datasets that
will be fed to the neural network during the training.

```shell
python generate_training_dataset.py
```

A script allows you to transform the raw dataset into `.ply` files, saved in `dataset/ply_files`. You can then read these files with CloudCompare or Meshlab for example to visualize the different point clouds.

```shell
python generate_ply.py
```

## Usage

First, you need to create a neural network model. To do so, use the following command. Add `--cuda` for GPU support. Skip this part if you want to resume a training or use an existing network.

```shell
python create_net.py --path networks/neural_net.pt
```

Then, you need to train the neural network. Add `--cuda` for GPU support. Type `--help` for more info.

```shell
python train_net.py --path networks/neural_net.pt
```
