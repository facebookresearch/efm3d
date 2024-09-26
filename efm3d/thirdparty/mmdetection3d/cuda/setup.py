# @lint-ignore-every LICENSELINT

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mmdet_iou3d",
    ext_modules=[
        CUDAExtension(
            "mmdet_iou3d",
            [
                "iou3d_kernel.cu",
                "iou3d.cpp",
                "sort_vert_kernel.cu",
                "sort_vert.cpp",
            ],
        )
    ],
    headers=[
        "iou3d.h",
        "sort_vert.h",
        "cuda_utils.h",
        "utils.h",
    ],
    cmdclass={"build_ext": BuildExtension},
)
