Installation
==================

.. _installation:

.. currentmodule:: tfts

This guide covers everything you need to install and set up TFTS (TensorFlow Time Series) for your environment.


Quick Installation
------------------

The fastest way to install TFTS is using pip:

.. code-block:: bash

   pip install tfts

This will install TFTS and its core dependencies.


Requirements
------------

System Requirements
~~~~~~~~~~~~~~~~~~~

**Minimum Requirements:**
   - Python 3.7 or higher
   - 4GB RAM (8GB+ recommended)
   - 2GB disk space

**Recommended:**
   - Python 3.8+
   - 16GB RAM for training large models
   - NVIDIA GPU with CUDA support (for GPU training)
   - 5GB+ disk space (including datasets and model checkpoints)

Dependencies
~~~~~~~~~~~~

TFTS requires the following core dependencies:

**Required:**
   - ``tensorflow >= 2.4.0``
   - ``numpy >= 1.19.0``
   - ``pandas >= 1.1.0``

**Optional but Recommended:**
   - ``scikit-learn >= 0.24.0`` (for preprocessing)
   - ``matplotlib >= 3.3.0`` (for visualization)
   - ``seaborn >= 0.11.0`` (for advanced plotting)


Installation Methods
--------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

Install the latest stable release from PyPI:

.. code-block:: bash

   pip install tfts


To upgrade to the latest version:

.. code-block:: bash

   pip install --upgrade tfts


From Source
~~~~~~~~~~~

For the latest development version, install from GitHub:

.. code-block:: bash

   git clone https://github.com/LongxingTan/Time-series-prediction.git
   cd Time-series-prediction
   pip install -e .

The ``-e`` flag installs in editable mode, allowing you to modify the source code.


For Development
~~~~~~~~~~~~~~~

If you plan to contribute or modify TFTS, install development dependencies:

.. code-block:: bash

   git clone https://github.com/LongxingTan/Time-series-prediction.git
   cd Time-series-prediction
   pip install -e ".[dev]"

This installs additional tools for testing, linting, and documentation.


Using Docker
~~~~~~~~~~~~

TFTS provides a Docker image with all dependencies pre-installed:

**Build the Docker Image:**

.. code-block:: bash

   docker build -f ./docker/Dockerfile -t tfts:latest .

**Run the Container:**

.. code-block:: bash

   docker run --rm -it \
       --init \
       --ipc=host \
       --network=host \
       --volume=$PWD:/app \
       --gpus all \
       tfts:latest /bin/bash

**For CPU-only:**

.. code-block:: bash

   docker run --rm -it \
       --init \
       --volume=$PWD:/app \
       tfts:latest /bin/bash


Environment-Specific Installation
----------------------------------

TensorFlow GPU Support
~~~~~~~~~~~~~~~~~~~~~~

For GPU acceleration, install TensorFlow with CUDA support:

**CUDA 11.2+ (Recommended):**

.. code-block:: bash

   pip install tensorflow[and-cuda]

**Manual CUDA Installation:**

1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Install cuDNN: https://developer.nvidia.com/cudnn
3. Install TensorFlow:

.. code-block:: bash

   pip install tensorflow-gpu
   pip install tfts

**Verify GPU Setup:**

.. code-block:: python

   import tensorflow as tf
   print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


Apple Silicon (M1/M2/M3)
~~~~~~~~~~~~~~~~~~~~~~~~

For macOS with Apple Silicon:

.. code-block:: bash

   # Install tensorflow-metal for GPU acceleration
   pip install tensorflow-macos tensorflow-metal
   pip install tfts

**Verify Metal Support:**

.. code-block:: python

   import tensorflow as tf
   print(tf.config.list_physical_devices())


TPU Support
~~~~~~~~~~~

For Google Cloud TPU:

.. code-block:: bash

   pip install cloud-tpu-client
   pip install tfts

**TPU Runtime Configuration:**

.. code-block:: python

   import tensorflow as tf

   resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
   tf.config.experimental_connect_to_cluster(resolver)
   tf.tpu.experimental.initialize_tpu_system(resolver)


Conda Environment
~~~~~~~~~~~~~~~~~

Create an isolated conda environment for TFTS:

.. code-block:: bash

   # Create environment
   conda create -n tfts python=3.9
   conda activate tfts

   # Install dependencies
   conda install tensorflow pandas numpy scikit-learn matplotlib
   pip install tfts


Virtual Environment
~~~~~~~~~~~~~~~~~~~

Using Python's built-in venv:

.. code-block:: bash

   # Create virtual environment
   python -m venv tfts-env

   # Activate (Linux/Mac)
   source tfts-env/bin/activate

   # Activate (Windows)
   tfts-env\Scripts\activate

   # Install TFTS
   pip install tfts


Troubleshooting
---------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**ImportError: No module named 'tensorflow'**

Solution: Install TensorFlow first:

.. code-block:: bash

   pip install tensorflow>=2.4


**CUDA version mismatch**

Solution: Ensure CUDA and cuDNN versions match TensorFlow requirements:

.. code-block:: bash

   # Check TensorFlow CUDA requirement
   python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"


**Memory errors during installation**

Solution: Install with no-cache option:

.. code-block:: bash

   pip install --no-cache-dir tfts


**Permission denied on Linux/Mac**

Solution: Use user installation:

.. code-block:: bash

   pip install --user tfts


Platform-Specific Issues
~~~~~~~~~~~~~~~~~~~~~~~~

**Windows:**
   - Use Anaconda/Miniconda for easier dependency management
   - Install Microsoft Visual C++ Redistributable if needed
   - Consider using WSL2 for Linux compatibility

**macOS:**
   - Install Xcode Command Line Tools: ``xcode-select --install``
   - Use Homebrew for system dependencies: ``brew install python``

**Linux:**
   - Install build essentials: ``sudo apt-get install build-essential``
   - For GPU: Install NVIDIA drivers and CUDA toolkit


Verification
------------

Verify Installation
~~~~~~~~~~~~~~~~~~~

Check that TFTS is installed correctly:

.. code-block:: python

   import tfts
   print(f"TFTS version: {tfts.__version__}")

   # Check available models
   from tfts import AutoConfig
   models = ['seq2seq', 'transformer', 'informer', 'autoformer']
   for model in models:
       config = AutoConfig.for_model(model)
       print(f"{model}: OK")


Run Test Suite
~~~~~~~~~~~~~~

Run the test suite to ensure everything works:

.. code-block:: bash

   # Install test dependencies
   pip install pytest pytest-cov

   # Run tests
   pytest tests/

   # Run with coverage
   pytest tests/ --cov=tfts


Quick Start Test
~~~~~~~~~~~~~~~~

Run a quick training test:

.. code-block:: python

   import tensorflow as tf
   import tfts
   from tfts import AutoConfig, AutoModel, KerasTrainer

   # Generate sample data
   train, valid = tfts.get_data('sine', train_length=24, predict_length=8)

   # Create and train model
   config = AutoConfig.for_model('seq2seq')
   model = AutoModel.from_config(config, predict_sequence_length=8)
   trainer = KerasTrainer(model)
   trainer.train(train, valid, epochs=2)

   print("✅ Installation successful!")


Updating TFTS
-------------

Stay Updated
~~~~~~~~~~~~

Keep TFTS up to date with the latest features and bug fixes:

.. code-block:: bash

   # Check current version
   pip show tfts

   # Update to latest version
   pip install --upgrade tfts

   # Update to specific version
   pip install --upgrade tfts==1.3.0


Development Builds
~~~~~~~~~~~~~~~~~~

For bleeding-edge features, install from the development branch:

.. code-block:: bash

   pip install git+https://github.com/LongxingTan/Time-series-prediction.git@master


Uninstallation
--------------

To remove TFTS:

.. code-block:: bash

   pip uninstall tfts

To completely remove including dependencies:

.. code-block:: bash

   pip uninstall tfts tensorflow pandas numpy scikit-learn matplotlib


Next Steps
----------

Now that you have TFTS installed:

1. **Quick Start:** Try the :doc:`quickstart` tutorial
2. **Learn the Basics:** Read :doc:`tutorials`
3. **Explore Models:** Check :doc:`models` documentation
4. **Prepare Data:** See :doc:`data_preparation` guide
5. **Train Models:** Follow :doc:`training` best practices


Getting Help
------------

If you encounter installation issues:

- 📖 Check the `FAQ <./faq.html>`_
- 💬 Ask in `GitHub Discussions <https://github.com/LongxingTan/Time-series-prediction/discussions>`_
- 🐛 Report bugs in `GitHub Issues <https://github.com/LongxingTan/Time-series-prediction/issues>`_
