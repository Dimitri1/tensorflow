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

## Installation

To install the current release for CPU-only:

```
pip install tensorflow
```

Use the GPU package for CUDA-enabled GPU cards:

```
pip install tensorflow-gpu
```

*See [Installing TensorFlow](https://www.tensorflow.org/install) for detailed
instructions, and how to build from source.*

People who are a little more adventurous can also try our nightly binaries:

**Nightly pip packages** * We are pleased to announce that TensorFlow now offers
nightly pip packages under the
[tf-nightly](https://pypi.python.org/pypi/tf-nightly) and
[tf-nightly-gpu](https://pypi.python.org/pypi/tf-nightly-gpu) project on PyPi.
Simply run `pip install tf-nightly` or `pip install tf-nightly-gpu` in a clean
environment to install the nightly TensorFlow build. We support CPU and GPU
packages on Linux, Mac, and Windows.

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

Learn more examples about how to do specific tasks in TensorFlow at the
[tutorials page of tensorflow.org](https://www.tensorflow.org/tutorials/).

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
**IBM s390x**                                                                     | [![Build Status](http://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_CI/badge/icon)](http://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_CI/)                                                             | TBA
**Linux ppc64le CPU** Nightly                                                     | [![Build Status](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Build/badge/icon)](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Build/)                                       | [Nightly](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Nightly_Artifact/)
**Linux ppc64le CPU** Stable Release                                              | [![Build Status](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Release_Build/badge/icon)](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Release_Build/)                       | [Release](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_CPU_Release_Build/)
**Linux ppc64le GPU** Nightly                                                     | [![Build Status](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Build/badge/icon)](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Build/)                                       | [Nightly](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Nightly_Artifact/)
**Linux ppc64le GPU** Stable Release                                              | [![Build Status](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Release_Build/badge/icon)](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Release_Build/)                       | [Release](https://powerci.osuosl.org/job/TensorFlow_PPC64LE_GPU_Release_Build/)
**Linux CPU with Intel® MKL-DNN** Nightly                                         | [![Build Status](https://tensorflow-ci.intel.com/job/tensorflow-mkl-linux-cpu/badge/icon)](https://tensorflow-ci.intel.com/job/tensorflow-mkl-linux-cpu/)                                     | [Nightly](https://tensorflow-ci.intel.com/job/tensorflow-mkl-build-whl-nightly/)
**Linux CPU with Intel® MKL-DNN** <br> **Supports Python 2.7, 3.4, 3.5, and 3.6** | [![Build Status](https://tensorflow-ci.intel.com/job/tensorflow-mkl-build-release-whl/badge/icon)](https://tensorflow-ci.intel.com/job/tensorflow-mkl-build-release-whl/lastStableBuild)      | [1.13.1 pypi](https://pypi.org/project/intel-tensorflow/)
**Red Hat® Enterprise Linux® 7.6 CPU & GPU** <br> Python 2.7, 3.6                 | [![Build Status](https://jenkins-tensorflow.apps.ci.centos.org/buildStatus/icon?job=tensorflow-rhel7-3.6&build=2)](https://jenkins-tensorflow.apps.ci.centos.org/job/tensorflow-rhel7-3.6/2/) | [1.13.1 pypi](https://tensorflow.pypi.thoth-station.ninja/index/)

## For more information

*   [TensorFlow Website](https://www.tensorflow.org)
*   [TensorFlow Tutorials](https://www.tensorflow.org/tutorials/)
*   [TensorFlow Model Zoo](https://github.com/tensorflow/models)
*   [TensorFlow Twitter](https://twitter.com/tensorflow)
*   [TensorFlow Blog](https://medium.com/tensorflow)
*   [TensorFlow Course at Stanford](https://web.stanford.edu/class/cs20si)
*   [TensorFlow Roadmap](https://www.tensorflow.org/community/roadmap)
*   [TensorFlow White Papers](https://www.tensorflow.org/about/bib)
*   [TensorFlow YouTube Channel](https://www.youtube.com/channel/UC0rqucBdTuFTjJiefW5t-IQ)
*   [TensorFlow Visualization Toolkit](https://github.com/tensorflow/tensorboard)

Learn more about the TensorFlow community at the [community page of tensorflow.org](https://www.tensorflow.org/community) for a few ways to participate.

## Build recomnentations

You need to compile these targets to use QAT : **//tensorflow/core/user_ops:quantemu.so**, **//tensorflow/core/user_ops:quantemu_op_py**, **//tensorflow/python:quantemu_ops**

#### Example (cpu only) :

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

See below a complete example of build procedure (used with docker):

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

## License

[Apache License 2.0](LICENSE)
