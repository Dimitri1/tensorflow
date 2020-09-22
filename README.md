<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_social.png">
</div>

-----------------


| **`Documentation`** |
|-----------------|
| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/) |

**TensorFlow** is an open source software library for numerical computation
using data flow graphs. The graph nodes represent mathematical operations, while
the graph edges represent the multidimensional data arrays (tensors) that flow
between them. This flexible architecture enables you to deploy computation to
one or more CPUs or GPUs in a desktop, server, or mobile device without
rewriting code. TensorFlow also includes
[TensorBoard](https://github.com/tensorflow/tensorboard), a data visualization
toolkit.

TensorFlow was originally developed by researchers and engineers
working on the Google Brain team within Google's Machine Intelligence Research
organization for the purposes of conducting machine learning and deep neural
networks research.  The system is general enough to be applicable in a wide
variety of other domains, as well.

TensorFlow provides stable Python and C APIs as well as non-guaranteed backwards
compatible API's for C++, Go, Java, JavaScript, and Swift.

Keep up to date with release announcements and security updates by
subscribing to
[announce@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce).

## Quant in tensorflow

This project consist of a custom version of tensorflow that was originaly
developped by Lumonk/tensorflow (https://github.com/Lumonk/tensorflow).

It is based on official tensorflow v1 (v1.14.0), and provides various extentions
to allow user to configure Quantization parameters at different levels (see **Usage** section).

## Build recomnentations

You need to compile these targets to use QAT : **//tensorflow/core/user_ops:quantemu.so**, **//tensorflow/core/user_ops:quantemu_op_py**, **//tensorflow/python:quantemu_ops**

```shell
bazel build --config=opt \
             --linkopt="-lrt" \
             --linkopt="-lm" \
             --host_linkopt="-lrt" \
             --host_linkopt="-lm" \
             --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
             //tensorflow/core/user_ops:quantemu.so

bazel build --config=opt \
             --linkopt="-lrt" \
             --linkopt="-lm" \
             --host_linkopt="-lrt" \
             --host_linkopt="-lm" \
             --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
             //tensorflow/core/user_ops:quantemu_op_py

bazel build --config=opt \
            --linkopt="-lrt" \
            --linkopt="-lm" \
            --host_linkopt="-lrt" \
            --host_linkopt="-lm" \
            --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
            //tensorflow/python:quantemu_ops

bazel build --config=opt \
            --linkopt="-lrt" \
            --linkopt="-lm" \
            --host_linkopt="-lrt" \
            --host_linkopt="-lm" \
            --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
            //tensorflow/tools/pip_package:build_pip_package
```

The upstream tensorflow version is v1.14.0, and it require numpy <1.15 (eg :1.14)

```shell
conda install --yes numpy==1.14
```

See below a complete example of build procedure (used with docker),
inspired by hadim/docker-tensorflow-builder (https://github.com/hadim/docker-tensorflow-builder):

```shell
#!/usr/bin/env bash
set -e
export PYTHON_VERSION=3.6
export TF_VERSION_GIT_TAG="v1.14.0"
export BAZEL_VERSION=0.24.1
export USE_GPU=0

export PATH="/conda/bin:/usr/bin:$PATH"

if [ "$USE_GPU" -eq "1" ]; then
  export CUDA_HOME="/usr/local/cuda"
  alias sudo=""
  source cuda.sh
  cuda.install $CUDA_VERSION $CUDNN_VERSION $NCCL_VERSION
  cd /
fi

# Set correct GCC version
GCC_VERSION="7"
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$GCC_VERSION 10
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-$GCC_VERSION 10
update-alternatives --set gcc "/usr/bin/gcc-$GCC_VERSION"
update-alternatives --set g++ "/usr/bin/g++-$GCC_VERSION"
gcc --version

# Install an appropriate Python environment
conda config --add channels conda-forge
conda create --yes -n tensorflow python==$PYTHON_VERSION
source activate tensorflow
conda install --yes wheel bazel==$BAZEL_VERSION
conda install --yes numpy==1.14
pip install keras-applications keras-preprocessing

# Compile TensorFlow from local mounted volume src/tensorflow
# You must provide TF_VERSION_GIT_TAG manually in top build.sh

TF_ROOT=/tensorflow
cd $TF_ROOT

##### SETUP ENV ######

# Python path options
export PYTHON_BIN_PATH=$(which python)
export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH=${TF_ROOT}/lib
export PYTHON_ARG=${TF_ROOT}/lib

# Compilation parameters
export TF_NEED_CUDA=0
export TF_NEED_GCP=1
export TF_CUDA_COMPUTE_CAPABILITIES=5.2,3.5
export TF_NEED_HDFS=1
export TF_NEED_OPENCL=0
export TF_NEED_JEMALLOC=1  # Need to be disabled on CentOS 6.6
export TF_ENABLE_XLA=0
export TF_NEED_VERBS=0
export TF_CUDA_CLANG=0
export TF_DOWNLOAD_CLANG=0
export TF_NEED_MKL=0
export TF_DOWNLOAD_MKL=0
export TF_NEED_MPI=0
export TF_NEED_S3=1
export TF_NEED_KAFKA=1
export TF_NEED_GDR=0
export TF_NEED_OPENCL_SYCL=0
export TF_SET_ANDROID_WORKSPACE=0
export TF_NEED_AWS=0
export TF_NEED_IGNITE=0
export TF_NEED_ROCM=0

# Compiler options
export GCC_HOST_COMPILER_PATH=$(which gcc)

# Here you can edit this variable to set any optimizations you want.
export CC_OPT_FLAGS="-march=native"

if [ "$USE_GPU" -eq "1" ]; then
  # Cuda parameters
  export CUDA_TOOLKIT_PATH=$CUDA_HOME
  export CUDNN_INSTALL_PATH=$CUDA_HOME
  export TF_CUDA_VERSION="$CUDA_VERSION"
  export TF_CUDNN_VERSION="$CUDNN_VERSION"
  export TF_NEED_CUDA=1
  export TF_NEED_TENSORRT=0
  export TF_NCCL_VERSION=$NCCL_VERSION
  export NCCL_INSTALL_PATH=$CUDA_HOME
  export NCCL_INSTALL_PATH=$CUDA_HOME

  # Those two lines are important for the linking step.
  export LD_LIBRARY_PATH="$CUDA_TOOLKIT_PATH/lib64:${LD_LIBRARY_PATH}"
  ldconfig
fi

# Compilation
./configure

if [ "$USE_GPU" -eq "1" ]; then

  bazel build --config=opt \
              --config=cuda \
              --linkopt="-lrt" \
              --linkopt="-lm" \
              --host_linkopt="-lrt" \
              --host_linkopt="-lm" \
              --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
              //tensorflow/tools/pip_package:build_pip_package

  PACKAGE_NAME=tensorflow-gpu
  SUBFOLDER_NAME="${TF_VERSION_GIT_TAG}-py${PYTHON_VERSION}-cuda${TF_CUDA_VERSION}-cudnn${TF_CUDNN_VERSION}"

else

  bazel build --config=opt \
               --linkopt="-lrt" \
               --linkopt="-lm" \
               --host_linkopt="-lrt" \
               --host_linkopt="-lm" \
               --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
               //tensorflow/core/user_ops:quantemu.so

  bazel build --config=opt \
               --linkopt="-lrt" \
               --linkopt="-lm" \
               --host_linkopt="-lrt" \
               --host_linkopt="-lm" \
               --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
               //tensorflow/core/user_ops:quantemu_op_py

  bazel build --config=opt \
              --linkopt="-lrt" \
              --linkopt="-lm" \
              --host_linkopt="-lrt" \
              --host_linkopt="-lm" \
              --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
              //tensorflow/python:quantemu_ops

  bazel build --config=opt \
              --linkopt="-lrt" \
              --linkopt="-lm" \
              --host_linkopt="-lrt" \
              --host_linkopt="-lm" \
              --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
              //tensorflow/tools/pip_package:build_pip_package

  PACKAGE_NAME=tensorflow
  SUBFOLDER_NAME="${TF_VERSION_GIT_TAG}-py${PYTHON_VERSION}"
fi

mkdir -p "/wheels/$SUBFOLDER_NAME"
bazel-bin/tensorflow/tools/pip_package/build_pip_package "/wheels/$SUBFOLDER_NAME" --project_name "$PACKAGE_NAME"
```

## Usage

You need to setup the quant variables before using
tensorflow. If you don't, quant operators won't be activated.

See below an example config for quant:

```shell
# Convolution 
ENABLE_QUANTOP_CONV=1
ENABLE_QUANTOP_CONV_GRAD=0
ENABLE_QUANTOP_CONV_WTGRAD=0

# BatchNormalization
ENABLE_QUANTOP_BNORM=1
ENABLE_QUANTOP_BNORM_NORM_ONLY=0
ENABLE_QUANTOP_BNORM_GRAD=0

# DENSE  
ENABLE_QUANTOP_DENSE=1
ENABLE_QUANTOP_DENSE_GRAD=0

# MATMUL 
ENABLE_QUANTOP_MATMUL=1
ENABLE_QUANTOP_MATMUL_GRAD=0

# MUL OP  
ENABLE_QUANTOP_MUL=1
ENABLE_QUANTOP_MUL_GRAD=0

# NON-LINEAR OPs
ENABLE_QUANTOP_SIGMOID=0
ENABLE_QUANTOP_SIGMOID_GRAD=0
ENABLE_QUANTOP_TANH=0
ENABLE_QUANTOP_TANH_GRAD=0

# Data Type Settings 
# INT=1, UINT=2, LOWP_FP=3, LOG2=4, POSIT=5, BFLOAT(RNE,STOCHASTIC)=6, MODFP16=7  
QUANTEMU_INPUT_DATA_TYPE=3
QUANTEMU_FILTER_DATA_TYPE=3
QUANTEMU_GRAD_DATA_TYPE=3
QUANTEMU_WTGRAD_DATA_TYPE=3

QUANTEMU_DENSE_DATA_TYPE=3 
QUANTEMU_BNORM_DATA_TYPE=3
QUANTEMU_MUL_DATA_TYPE=3
QUANTEMU_TANH_DATA_TYPE=3
QUANTEMU_SIGMOID_DATA_TYPE=3

# only used by LOWP_FP, POSIT and BLOCK_FP types 
QUANTEMU_EXPBITS=5

# Rounding modes  
# Truncate (no rounding)=0, Round to Nearest Even(RNE)=1, STOCHASTIC_ROUNDING=2  
QUANTEMU_RMODE_INPUTS=2
QUANTEMU_RMODE_FILTERS=1
QUANTEMU_RMODE_GRADS=2
QUANTEMU_RMODE_WTGRADS=1
QUANTEMU_BNORM_RMODE_INPUTS=1
QUANTEMU_BNORM_RMODE_GRADS=1

# Precision Settings 
QUANTEMU_FIRST_LAYER_PRECISION=16

QUANTEMU_PRECISION_CONV_INPUTS=8
QUANTEMU_PRECISION_CONV_FILTERS=8
QUANTEMU_PRECISION_CONV_GRADS=8
QUANTEMU_PRECISION_CONV_WTGRADS=8

QUANTEMU_PRECISION_DENSE_INPUTS=8
QUANTEMU_PRECISION_DENSE_FILTERS=8
QUANTEMU_PRECISION_DENSE_GRADS=8

QUANTEMU_PRECISION_BNORM_INPUTS=16
QUANTEMU_PRECISION_BNORM_GRADS=16
QUANTEMU_PRECISION_MATMUL_INPUTS=8
QUANTEMU_PRECISION_MATMUL_FILTERS=8
QUANTEMU_PRECISION_MATMUL_GRADS=8

QUANTEMU_PRECISION_MUL_INPUTS=8
QUANTEMU_PRECISION_MUL_GRADS=8
QUANTEMU_PRECISION_SIGMOID_INPUTS=3 
QUANTEMU_PRECISION_SIGMOID_GRADS=3 
QUANTEMU_PRECISION_TANH_INPUTS=3 
QUANTEMU_PRECISION_TANH_GRADS=3 

# Buffer Copy Settings 
# Make a Copy while Quantizing 
QUANTEMU_ALLOCATE_COPY_INPUTS=0
QUANTEMU_ALLOCATE_COPY_FILTERS=0
QUANTEMU_ALLOCATE_COPY_GRADS=0

# FGQ Settings 
# NOBLOCK=0, BLOCK_C=1, BLOCK_CHW=2 
QUANTEMU_CBLOCK_TYPE_CONV_INPUTS=0
QUANTEMU_CBLOCK_TYPE_CONV_FILTERS=0
QUANTEMU_CBLOCK_TYPE_CONV_GRADS=0
QUANTEMU_CBLOCK_TYPE_CONV_WTGRADS=0

QUANTEMU_CBLOCK_TYPE_BNORM_INPUTS=0
QUANTEMU_CBLOCK_TYPE_BNORM_GRADS=0

QUANTEMU_CBLOCK_SIZE_INPUTS=2048
QUANTEMU_CBLOCK_SIZE_FILTER=2048
QUANTEMU_CBLOCK_SIZE_GRAD=2048
QUANTEMU_CBLOCK_SIZE_WTGRAD=2048

export ENABLE_QUANTOP_CONV
export ENABLE_QUANTOP_CONV_GRAD
export ENABLE_QUANTOP_CONV_WTGRAD
export QUANTEMU_WTGRAD_DATA_TYPE

export QUANTEMU_MUL_DATA_TYPE
export QUANTEMU_BNORM_DATA_TYPE
export QUANTEMU_DENSE_DATA_TYPE
export QUANTEMU_TANH_DATA_TYPE
export QUANTEMU_SIGMOID_DATA_TYPE

export QUANTEMU_RMODE_WTGRADS
export QUANTEMU_PRECISION_CONV_WTGRADS
export QUANTEMU_CBLOCK_TYPE_CONV_WTGRADS
export QUANTEMU_CBLOCK_SIZE_WTGRAD

export ENABLE_QUANTOP_BNORM
export ENABLE_QUANTOP_BNORM_NORM_ONLY
export ENABLE_QUANTOP_BNORM_GRAD
export ENABLE_QUANTOP_DENSE
export ENABLE_QUANTOP_DENSE_GRAD

export QUANTEMU_BNORM_RMODE_INPUTS
export QUANTEMU_BNORM_RMODE_GRADS

export ENABLE_QUANTOP_MATMUL
export ENABLE_QUANTOP_MATMUL_GRAD
export ENABLE_QUANTOP_MUL
export ENABLE_QUANTOP_MUL_GRAD

export ENABLE_QUANTOP_SIGMOID
export ENABLE_QUANTOP_SIGMOID_GRAD
export ENABLE_QUANTOP_TANH
export ENABLE_QUANTOP_TANH_GRAD

export QUANTEMU_INPUT_DATA_TYPE
export QUANTEMU_FILTER_DATA_TYPE
export QUANTEMU_GRAD_DATA_TYPE
export QUANTEMU_EXPBITS          
export QUANTEMU_RMODE_INPUTS
export QUANTEMU_RMODE_FILTERS
export QUANTEMU_RMODE_GRADS
export QUANTEMU_FIRST_LAYER_PRECISION
export QUANTEMU_PRECISION_CONV_INPUTS
export QUANTEMU_PRECISION_CONV_FILTERS
export QUANTEMU_PRECISION_CONV_GRADS
export QUANTEMU_PRECISION_DENSE_INPUTS
export QUANTEMU_PRECISION_DENSE_FILTERS
export QUANTEMU_PRECISION_DENSE_GRADS
export QUANTEMU_PRECISION_BNORM_INPUTS
export QUANTEMU_PRECISION_BNORM_GRADS
export QUANTEMU_PRECISION_MATMUL_INPUTS
export QUANTEMU_PRECISION_MATMUL_FILTERS
export QUANTEMU_PRECISION_MATMUL_GRADS
export QUANTEMU_PRECISION_MUL_INPUTS
export QUANTEMU_PRECISION_MUL_GRADS

export QUANTEMU_PRECISION_SIGMOID_INPUTS
export QUANTEMU_PRECISION_SIGMOID_GRADS
export QUANTEMU_PRECISION_TANH_INPUTS
export QUANTEMU_PRECISION_TANH_GRADS 

export QUANTEMU_ALLOCATE_COPY_INPUTS
export QUANTEMU_ALLOCATE_COPY_FILTERS
export QUANTEMU_ALLOCATE_COPY_GRADS
export QUANTEMU_CBLOCK_TYPE_CONV_INPUTS
export QUANTEMU_CBLOCK_TYPE_CONV_FILTERS
export QUANTEMU_CBLOCK_TYPE_CONV_GRADS
export QUANTEMU_CBLOCK_TYPE_BNORM_INPUTS
export QUANTEMU_CBLOCK_TYPE_BNORM_GRADS
export QUANTEMU_CBLOCK_SIZE_INPUTS
export QUANTEMU_CBLOCK_SIZE_FILTER
export QUANTEMU_CBLOCK_SIZE_GRAD
```

You can copy in a file, and source this file
before calling your tensorflow application.




## License

[Apache License 2.0](LICENSE)
