Point_Cloud_Labelling

## Prerequisites

You need to have Python 3.7 (at least) installed on your computer. To run the program, commands are the following.

```shell
git clone <repo.git>
pip install -r requirements.txt
```

## Dataset

Download the dataset with on `wget` or `curl` and unzip it with `unzip` or another program.

```shell
curl -o dataset/dataset.zip https://www.vision.rwth-aachen.de/media/resource_files/vkitti3d_dataset_v1.0.zip
wget -O dataset/dataset.zip https://www.vision.rwth-aachen.de/media/resource_files/vkitti3d_dataset_v1.0.zip
unzip dataset/dataset.zip -d dataset/numpy_files
```

A script allows you to transform the raw dataset into `.ply` files, saved in `dataset/ply_files`. You can then read these files in CloudCompare or Meshlab for example to visualize the point clouds.

```shell
python generate_ply.py
```
