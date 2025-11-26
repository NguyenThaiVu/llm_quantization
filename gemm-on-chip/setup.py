import os
import pathlib
os.environ["CUDA_HOME"] = "/home/tnguyen10/cuda-12.1"
os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:" + os.environ["PATH"]
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

include_dirs=["/home/tnguyen10/Desktop/deep_learning_research/llm_quantization/gemm-on-chip/cutlass/include",\
                "/home/tnguyen10/Desktop/deep_learning_research/llm_quantization/gemm-on-chip/cutlass/tools/util/include"]
cuda  = os.environ.get("CUDA_HOME", "")

setup(
    name="gemm_cutlass",
    ext_modules=[
        CUDAExtension(
            name="gemm_cutlass",
            sources=["gemm_cutlass.cu"],
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-gencode=arch=compute_80,code=sm_80",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)