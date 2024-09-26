# Installation

## Inference

### Install via pip

The core library is written in
[pytorch](https://pytorch.org/get-started/locally/). Make sure you have
Python>=3.9, then install the dependencies using `pip`. A few notable packages
include `webdataset`, `vrs` and `trimesh`.

```
pip install -r requirements.txt
```

The packages in `requirements.txt` are needed for the basic functionalities of
EFM3D, such as running the example model inference to see 3D object detection
and surface reconstruction on a [vrs](https://facebookresearch.github.io/vrs/)
sequence. Note that it doesn't include the instructions for setting up
torch-related libs, which are required to run the library. Additional
dependencies are needed for training and eval.

### Install using conda

Conda installation can be found at
[here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
This is the recommended approach to manage dependencies. The runtime
dependencies can be installed by running

```
conda env create --file=environment.yml
conda activate efm3d
```

## Additional dependencies

Install the additional dependencies by running

```
pip install -r requirements-extra.txt
```

You may need to install `gcc` and `python3-dev` first. Check out the
[pytorch3d installation](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
guide for details.

These dependencies enable training with
[Aria Training and Evaluation Kit (ATEK)](https://github.com/facebookresearch/atek),
and 3D object evaluation. Note that installing `pytorch3d` could take a long
time, which is needed by object tracker and 3D object evaluation.

### Training

**Important**: For training, we also need to built a CUDA kernel from
[mmdetection3d](https://github.com/open-mmlab/mmdetection3d). Compile the CUDA
kernel of the IoU3d loss by running the following commands, which requires the
installation of
[CUDA dev toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

```
pip install setuptools==69.5.1
cd efm3d/thirdparty/mmdetection3d/cuda/
python setup.py install
```
