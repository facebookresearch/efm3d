# Installation

We provide two ways to install the dependencies of EFM3D. We recommend using miniconda to manage the dependencies, which
also provide a easy setup to for all the additional dependencies listed in `requirements.txt` and `requirements-extra.txt`.

## Install using conda (recommended)

First install [miniconda](https://docs.anaconda.com/miniconda/#quick-command-line-install),
then run the following commands under the `<EFM3D_DIR>` root directory

```
conda env create --file=environment.yml
conda activate efm3d

cd efm3d/thirdparty/mmdetection3d/cuda/
python setup.py install
```

The commands will first create a conda environment named `efm3d`, and then build the
third-party CUDA kernel required for training.

## Install via pip

Make sure you have
Python>=3.9, then install the dependencies using `pip`.
The packages in `requirements.txt` are needed for the basic functionalities of
EFM3D, such as running the example model inference to see 3D object detection
and surface reconstruction on a [vrs](https://facebookresearch.github.io/vrs/)
sequence.

```
pip install -r requirements.txt
```

Additional dependencies in `requirements-extra.txt` are needed for training and eval.

```
pip install -r requirements-extra.txt
```

**Important**: For training, we also need to built a CUDA kernel from
[mmdetection3d](https://github.com/open-mmlab/mmdetection3d). Compile the CUDA
kernel of the IoU3d loss by running the following commands, which requires the
installation of
[CUDA dev toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

```
cd efm3d/thirdparty/mmdetection3d/cuda/
python setup.py install
```
