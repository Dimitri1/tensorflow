<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_social.png">
</div>

| **`Documentation`** |
|-----------------|
| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/) |

[TensorFlow](https://www.tensorflow.org/) is an end-to-end open source platform
for machine learning. It has a comprehensive, flexible ecosystem of
[tools](https://www.tensorflow.org/resources/tools),
[libraries](https://www.tensorflow.org/resources/libraries-extensions), and
[community](https://www.tensorflow.org/community) resources that lets
researchers push the state-of-the-art in ML and developers easily build and
deploy ML powered applications.

TensorFlow was originally developed by researchers and engineers working on the
Google Brain team within Google's Machine Intelligence Research organization for
the purposes of conducting machine learning and deep neural networks research.
The system is general enough to be applicable in a wide variety of other
domains, as well.

TensorFlow provides stable [Python](https://www.tensorflow.org/api_docs/python)
and [C++](https://www.tensorflow.org/api_docs/cc) APIs, as well as
non-guaranteed backwards compatible API for
[other languages](https://www.tensorflow.org/api_docs).

Keep up-to-date with release announcements and security updates by subscribing
to
[announce@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce).
See all the [mailing lists](https://www.tensorflow.org/community/forums).

## Install

See the [TensorFlow install guide](https://www.tensorflow.org/install) for the
[pip package](https://www.tensorflow.org/install/pip), to
[enable GPU support](https://www.tensorflow.org/install/gpu), use a
[Docker container](https://www.tensorflow.org/install/docker), and
[build from source](https://www.tensorflow.org/install/source).

To install the current release for CPU-only:

```
$ pip install tensorflow
```

Use the GPU package for
[CUDA-enabled GPU cards](https://www.tensorflow.org/install/gpu):

```
$ pip install tensorflow-gpu
```

*Nightly binaries are available for testing using the
[tf-nightly](https://pypi.python.org/pypi/tf-nightly) and
[tf-nightly-gpu](https://pypi.python.org/pypi/tf-nightly-gpu) packages on PyPi.*

#### *Try your first TensorFlow program*

```shell
$ python
```

```python
>>> import tensorflow as tf
>>> tf.enable_eager_execution()
>>> tf.add(1, 2).numpy()
3
>>> hello = tf.constant('Hello, TensorFlow!')
>>> hello.numpy()
'Hello, TensorFlow!'
```

For more examples, see the
[TensorFlow tutorials](https://www.tensorflow.org/tutorials/).

## Contribution guidelines

**If you want to contribute to TensorFlow, be sure to review the [contribution
guidelines](CONTRIBUTING.md). This project adheres to TensorFlow's
[code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.**

**We use [GitHub issues](https://github.com/tensorflow/tensorflow/issues) for
tracking requests and bugs, please see
[TensorFlow Discuss](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss)
for general questions and discussion, and please direct specific questions to
[Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow).**

The TensorFlow project strives to abide by generally accepted best practices in open-source software development:

[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/1486/badge)](https://bestpractices.coreinfrastructure.org/projects/1486)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)

## Continuous build status

### Official Builds

| Build Type      | Status | Artifacts |
| ---             | ---    | ---       |
| **Linux CPU**   | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-cc.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-cc.html) | [pypi](https://pypi.org/project/tf-nightly/) |
| **Linux GPU**   | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-gpu-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-gpu-py3.html) | [pypi](https://pypi.org/project/tf-nightly-gpu/) |
| **Linux XLA**   | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-xla.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-xla.html) | TBA |
| **MacOS**       | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/macos-py2-cc.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/macos-py2-cc.html) | [pypi](https://pypi.org/project/tf-nightly/) |
| **Windows CPU** | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-cpu.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-cpu.html) | [pypi](https://pypi.org/project/tf-nightly/) |
| **Windows GPU** | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-gpu.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-gpu.html) | [pypi](https://pypi.org/project/tf-nightly-gpu/) |
| **Android**     | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/android.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/android.html) | [![Download](https://api.bintray.com/packages/google/tensorflow/tensorflow/images/download.svg)](https://bintray.com/google/tensorflow/tensorflow/_latestVersion) |
| **Raspberry Pi 0 and 1** | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi01-py2.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi01-py2.html) [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi01-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi01-py3.html) | [Py2](https://storage.googleapis.com/tensorflow-nightly/tensorflow-1.10.0-cp27-none-linux_armv6l.whl) [Py3](https://storage.googleapis.com/tensorflow-nightly/tensorflow-1.10.0-cp34-none-linux_armv6l.whl) |
| **Raspberry Pi 2 and 3** | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi23-py2.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi23-py2.html) [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi23-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi23-py3.html) | [Py2](https://storage.googleapis.com/tensorflow-nightly/tensorflow-1.10.0-cp27-none-linux_armv7l.whl) [Py3](https://storage.googleapis.com/tensorflow-nightly/tensorflow-1.10.0-cp34-none-linux_armv7l.whl) |


### Community Supported Builds

Build Type                                                                        | Status                                                                                                                                                                                        | Artifacts
--------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------
**Linux AMD ROCm GPU** Nightly                                                    | [![Build Status](http://ml-ci.amd.com:21096/job/tensorflow-rocm-nightly/badge/icon)](http://ml-ci.amd.com:21096/job/tensorflow-rocm-nightly)                                                  | [Nightly](http://ml-ci.amd.com:21096/job/tensorflow-rocm-nightly/lastSuccessfulBuild/)
**Linux AMD ROCm GPU** Stable Release                                             | [![Build Status](http://ml-ci.amd.com:21096/job/tensorflow-rocm-release/badge/icon)](http://ml-ci.amd.com:21096/job/tensorflow-rocm-release/)                                                 | [Release](http://ml-ci.amd.com:21096/job/tensorflow-rocm-release/lastSuccessfulBuild/)
**Linux s390x** Nightly                                                           | [![Build Status](http://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_CI/badge/icon)](http://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_CI/)                                                             | [Nightly](http://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_CI/)
**Linux s390x CPU** Stable Release                                                | [![Build Status](http://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_Release_Build/badge/icon)](https://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_Release_Build/)                                      | [Release](https://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_Release_Build/)
**Linux ppc64le CPU** Nightly                                                     | [![Build Status](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Build/badge/icon)](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Build/)                                       | [Nightly](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Nightly_Artifact/)
**Linux ppc64le CPU** Stable Release                                              | [![Build Status](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Release_Build/badge/icon)](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Release_Build/)                       | [Release](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Release_Build/)
**Linux ppc64le GPU** Nightly                                                     | [![Build Status](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Build/badge/icon)](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Build/)                                       | [Nightly](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Nightly_Artifact/)
**Linux ppc64le GPU** Stable Release                                              | [![Build Status](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Release_Build/badge/icon)](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Release_Build/)                       | [Release](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Release_Build/)
**Linux CPU with Intel® MKL-DNN** Nightly                                         | [![Build Status](https://tensorflow-ci.intel.com/job/tensorflow-mkl-linux-cpu/badge/icon)](https://tensorflow-ci.intel.com/job/tensorflow-mkl-linux-cpu/)                                     | [Nightly](https://tensorflow-ci.intel.com/job/tensorflow-mkl-build-whl-nightly/)
**Linux CPU with Intel® MKL-DNN** <br> **Supports Python 2.7, 3.4, 3.5, and 3.6** | [![Build Status](https://tensorflow-ci.intel.com/job/tensorflow-mkl-build-release-whl/badge/icon)](https://tensorflow-ci.intel.com/job/tensorflow-mkl-build-release-whl/lastStableBuild)      | [1.13.1 pypi](https://pypi.org/project/intel-tensorflow/)
**Red Hat® Enterprise Linux® 7.6 CPU & GPU** <br> Python 2.7, 3.6                 | [![Build Status](https://jenkins-tensorflow.apps.ci.centos.org/buildStatus/icon?job=tensorflow-rhel7-3.6&build=2)](https://jenkins-tensorflow.apps.ci.centos.org/job/tensorflow-rhel7-3.6/2/) | [1.13.1 pypi](https://tensorflow.pypi.thoth-station.ninja/index/)

## Resources

*   [TensorFlow.org](https://www.tensorflow.org)
*   [TensorFlow tutorials](https://www.tensorflow.org/tutorials/)
*   [TensorFlow official models](https://github.com/tensorflow/models/tree/master/official)
*   [TensorFlow examples](https://github.com/tensorflow/examples)
*   [TensorFlow in Practice from Coursera](https://www.coursera.org/specializations/tensorflow-in-practice)
*   [TensorFlow blog](https://blog.tensorflow.org)
*   [TensorFlow Twitter](https://twitter.com/tensorflow)
*   [TensorFlow YouTube](https://www.youtube.com/channel/UC0rqucBdTuFTjJiefW5t-IQ)
*   [TensorFlow roadmap](https://www.tensorflow.org/community/roadmap)
*   [TensorFlow white papers](https://www.tensorflow.org/about/bib)
*   [TensorBoard visualization toolkit](https://github.com/tensorflow/tensorboard)

Learn more about the
[TensorFlow community](https://www.tensorflow.org/community) and how to
[contribute](https://www.tensorflow.org/community/contribute).

## License

[Apache License 2.0](LICENSE)


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
