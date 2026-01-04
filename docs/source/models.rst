Models
======

.. _models:

.. currentmodule:: tfts

TFTS provides a comprehensive collection of state-of-the-art deep learning models for time series analysis. All models are accessible through a unified API and can be easily configured, trained, and deployed.


Model Overview
--------------

Available Models
~~~~~~~~~~~~~~~~

TFTS supports 20+ model architectures, organized by category:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Model
     - Type
     - Best For
   * - ``seq2seq``
     - RNN-based
     - General purpose, interpretable baselines
   * - ``rnn``
     - RNN-based
     - Simple sequence modeling, quick prototyping
   * - ``deep_ar``
     - RNN-based
     - Probabilistic forecasting with uncertainty
   * - ``tcn``
     - CNN-based
     - Long-range dependencies, fast inference
   * - ``wavenet``
     - CNN-based
     - High-frequency signals, audio-like data
   * - ``unet``
     - CNN-based
     - Sequence-to-sequence with skip connections
   * - ``transformer``
     - Transformer
     - Complex patterns, long-term dependencies
   * - ``bert``
     - Transformer
     - Representation learning, pre-training
   * - ``informer``
     - Transformer
     - Very long sequences (1000+ steps)
   * - ``autoformer``
     - Transformer
     - Seasonal data, decomposition tasks
   * - ``tft``
     - Transformer
     - Interpretable attention, multiple inputs
   * - ``patch_tst``
     - Transformer
     - Efficient training on long sequences
   * - ``itransformer``
     - Transformer
     - Multivariate with variable relationships
   * - ``nbeats``
     - Specialized
     - Interpretable basis expansion
   * - ``dlinear``
     - Specialized
     - Simple, fast baseline with decomposition
   * - ``rwkv``
     - Specialized
     - Linear attention, efficient memory
   * - ``diffusion``
     - Specialized
     - Uncertainty quantification, generation
   * - ``tide``
     - Specialized
     - Dense encoder, simple architecture
   * - ``gpt``
     - Specialized
     - Autoregressive generation


Model Selection Guide
~~~~~~~~~~~~~~~~~~~~~

Choose the right model based on your requirements:

**For Quick Prototyping:**
   Start with ``rnn`` or ``dlinear`` for fast iteration and baseline performance.

**For Long Sequences (>500 steps):**
   Use ``informer``, ``patch_tst``, or ``autoformer`` with efficient attention mechanisms.

**For Interpretability:**
   Choose ``nbeats`` (interpretable basis), ``tft`` (attention visualization), or ``dlinear`` (linear decomposition).

**For Probabilistic Forecasting:**
   Use ``deep_ar`` or ``diffusion`` for uncertainty quantification.

**For Multivariate Data:**
   Consider ``itransformer`` (treats variables as tokens) or ``tft`` (handles multiple inputs).

**For Computational Efficiency:**
   Choose ``tcn`` (fast convolutions), ``dlinear`` (simple linear), or ``rwkv`` (linear attention).


Detailed Model Descriptions
----------------------------

RNN-Based Models
~~~~~~~~~~~~~~~~

Seq2seq (Sequence-to-Sequence)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Classic encoder-decoder architecture with attention mechanism.

**Architecture:**
   - Encoder: LSTM/GRU cells that compress input sequence into context vectors
   - Decoder: LSTM/GRU cells that generate output sequence with attention
   - Attention: Bahdanau or Luong-style attention mechanism

**Best For:**
   - General-purpose forecasting
   - Interpretable attention weights
   - Baseline comparisons

**Example:**

.. code-block:: python

   from tfts import AutoConfig, AutoModel

   config = AutoConfig.for_model('seq2seq')
   config.rnn_type = 'lstm'  # or 'gru'
   config.rnn_size = 128
   config.attention_sizes = 128

   model = AutoModel.from_config(config, predict_sequence_length=24)

**Key Parameters:**
   - ``rnn_type``: Choose between 'lstm' or 'gru'
   - ``rnn_size``: Hidden state dimension (default: 128)
   - ``attention_sizes``: Attention mechanism dimension
   - ``num_layers``: Number of stacked RNN layers

**References:**
   - Sutskever et al. "Sequence to Sequence Learning with Neural Networks" (NeurIPS 2014)
   - Bahdanau et al. "Neural Machine Translation by Jointly Learning to Align and Translate" (ICLR 2015)


DeepAR
^^^^^^

Probabilistic forecasting model using autoregressive RNN.

**Architecture:**
   - LSTM-based encoder
   - Parametric output distributions (Gaussian, Student-t, etc.)
   - Monte Carlo sampling for uncertainty quantification

**Best For:**
   - Probabilistic forecasting with confidence intervals
   - Scenarios requiring uncertainty quantification
   - Multiple related time series

**Example:**

.. code-block:: python

   from tfts import AutoConfig, AutoModel

   config = AutoConfig.for_model('deep_ar')
   config.rnn_size = 64
   config.num_samples = 100  # MC samples

   model = AutoModel.from_config(config, predict_sequence_length=24)

**References:**
   - Salinas et al. "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks" (2020)


CNN-Based Models
~~~~~~~~~~~~~~~~

TCN (Temporal Convolutional Network)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dilated causal convolutions for sequence modeling.

**Architecture:**
   - Dilated causal convolutions with exponentially increasing dilation rates
   - Residual connections for gradient flow
   - Weight normalization for stable training

**Best For:**
   - Long-range dependencies
   - Fast parallel training and inference
   - Real-time applications

**Example:**

.. code-block:: python

   from tfts import AutoConfig, AutoModel

   config = AutoConfig.for_model('tcn')
   config.filters = 64
   config.kernel_size = 3
   config.num_blocks = 3
   config.dropout = 0.1

   model = AutoModel.from_config(config, predict_sequence_length=24)

**Key Parameters:**
   - ``filters``: Number of convolutional filters
   - ``kernel_size``: Convolution kernel size
   - ``num_blocks``: Number of dilated residual blocks
   - ``dilation_rate``: Exponential dilation factor

**References:**
   - Bai et al. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks" (2018)


WaveNet
^^^^^^^

Deep generative model with dilated causal convolutions.

**Architecture:**
   - Stacked dilated causal convolutional layers
   - Gated activation functions (tanh and sigmoid)
   - Skip connections aggregating information from all layers

**Best For:**
   - High-frequency signals
   - Audio and sensor data
   - Complex temporal patterns

**Example:**

.. code-block:: python

   from tfts import AutoConfig, AutoModel

   config = AutoConfig.for_model('wavenet')
   config.filters = 32
   config.num_blocks = 2
   config.num_layers = 10

   model = AutoModel.from_config(config, predict_sequence_length=24)

**References:**
   - van den Oord et al. "WaveNet: A Generative Model for Raw Audio" (2016)


Transformer-Based Models
~~~~~~~~~~~~~~~~~~~~~~~~

Transformer
^^^^^^^^^^^

Standard Transformer architecture adapted for time series.

**Architecture:**
   - Multi-head self-attention mechanism
   - Position-wise feed-forward networks
   - Layer normalization and residual connections
   - Positional encoding for temporal information

**Best For:**
   - General-purpose time series modeling
   - Learning complex patterns
   - Transfer learning applications

**Example:**

.. code-block:: python

   from tfts import AutoConfig, AutoModel

   config = AutoConfig.for_model('transformer')
   config.hidden_size = 128
   config.num_layers = 3
   config.num_attention_heads = 8
   config.attention_probs_dropout_prob = 0.1

   model = AutoModel.from_config(config, predict_sequence_length=24)

**Key Parameters:**
   - ``hidden_size``: Model dimension
   - ``num_layers``: Number of encoder/decoder layers
   - ``num_attention_heads``: Parallel attention heads
   - ``ffn_intermediate_size``: Feed-forward network hidden size

**References:**
   - Vaswani et al. "Attention Is All You Need" (NeurIPS 2017)


Informer
^^^^^^^^

Efficient Transformer for long sequence time series forecasting.

**Architecture:**
   - ProbSparse self-attention: O(L log L) complexity vs O(L²)
   - Self-attention distilling for reduced memory
   - Generative decoder for one-forward prediction

**Best For:**
   - Very long sequences (1000+ time steps)
   - Memory-constrained environments
   - Long-term forecasting (LSTF) tasks

**Example:**

.. code-block:: python

   from tfts import AutoConfig, AutoModel

   config = AutoConfig.for_model('informer')
   config.hidden_size = 256
   config.num_layers = 3
   config.num_attention_heads = 8
   config.factor = 5  # ProbSparse factor

   model = AutoModel.from_config(config, predict_sequence_length=96)

**Key Parameters:**
   - ``factor``: Sampling factor for ProbSparse attention (higher = more efficient)
   - ``distil``: Enable attention distilling

**References:**
   - Zhou et al. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (AAAI 2021)


Autoformer
^^^^^^^^^^

Transformer with Auto-Correlation mechanism and decomposition.

**Architecture:**
   - Auto-correlation replaces self-attention for periodic patterns
   - Series decomposition (trend + seasonal) at each layer
   - Aggregates periodic components automatically

**Best For:**
   - Seasonal time series
   - Data with clear periodic patterns
   - Long-term forecasting

**Example:**

.. code-block:: python

   from tfts import AutoConfig, AutoModel

   config = AutoConfig.for_model('autoformer')
   config.hidden_size = 128
   config.num_layers = 2
   config.moving_avg = 25  # Window for decomposition

   model = AutoModel.from_config(config, predict_sequence_length=96)

**References:**
   - Wu et al. "Autoformer: Decomposition Transformers with Auto-Correlation" (NeurIPS 2021)


Temporal Fusion Transformer (TFT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Attention-based model with interpretable multi-horizon forecasting.

**Architecture:**
   - Variable selection network for feature importance
   - LSTM for local processing
   - Multi-head attention for temporal relationships
   - Quantile outputs for uncertainty

**Best For:**
   - Multiple input features (static + dynamic)
   - Interpretable attention weights
   - Multi-horizon probabilistic forecasting

**Example:**

.. code-block:: python

   from tfts import AutoConfig, AutoModel

   config = AutoConfig.for_model('tft')
   config.hidden_size = 160
   config.num_attention_heads = 4

   model = AutoModel.from_config(config, predict_sequence_length=24)

**References:**
   - Lim et al. "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (2021)


PatchTST
^^^^^^^^

Patch-based Transformer for efficient time series modeling.

**Architecture:**
   - Divides time series into patches (sub-sequences)
   - Treats patches as tokens for Transformer
   - Channel independence for multivariate modeling

**Best For:**
   - Long sequences with efficient training
   - Multivariate time series
   - Transfer learning across datasets

**Example:**

.. code-block:: python

   from tfts import AutoConfig, AutoModel

   config = AutoConfig.for_model('patch_tst')
   config.patch_size = 16  # Patch length
   config.hidden_size = 128
   config.num_layers = 3

   model = AutoModel.from_config(config, predict_sequence_length=96)

**References:**
   - Nie et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023)


iTransformer
^^^^^^^^^^^^

Inverted Transformer treating variates as tokens.

**Architecture:**
   - Inverts the role of time and variables
   - Each variable becomes a token
   - Learns relationships between variables explicitly

**Best For:**
   - Multivariate forecasting with variable interactions
   - Learning cross-variate dependencies
   - High-dimensional time series

**Example:**

.. code-block:: python

   from tfts import AutoConfig, AutoModel

   config = AutoConfig.for_model('itransformer')
   config.hidden_size = 128
   config.num_layers = 3

   model = AutoModel.from_config(config, predict_sequence_length=96)

**References:**
   - Liu et al. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (2023)


Specialized Models
~~~~~~~~~~~~~~~~~~

N-BEATS
^^^^^^^

Neural Basis Expansion Analysis for interpretable forecasting.

**Architecture:**
   - Stack of fully-connected blocks
   - Each block produces basis expansion coefficients
   - Interpretable (trend + seasonality) or generic versions
   - Doubly residual stacking for hierarchical patterns

**Best For:**
   - Interpretable univariate forecasting
   - M4 competition-style problems
   - When basis expansion is meaningful

**Example:**

.. code-block:: python

   from tfts import AutoConfig, AutoModel

   config = AutoConfig.for_model('nbeats')
   config.num_blocks = 3
   config.stack_types = ['trend', 'seasonality']
   config.num_layers_per_block = 4

   model = AutoModel.from_config(config, predict_sequence_length=24)

**References:**
   - Oreshkin et al. "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting" (ICLR 2020)


DLinear
^^^^^^^

Simple linear model with seasonal-trend decomposition.

**Architecture:**
   - Decomposes series into trend and seasonal components
   - Applies separate linear layers to each component
   - Extremely simple yet effective baseline

**Best For:**
   - Simple baselines
   - Fast training and inference
   - When complexity isn't justified

**Example:**

.. code-block:: python

   from tfts import AutoConfig, AutoModel

   config = AutoConfig.for_model('dlinear')
   config.moving_avg = 25  # Decomposition window
   config.channels = 1  # Number of features

   model = AutoModel.from_config(config, predict_sequence_length=96)

**References:**
   - Zeng et al. "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)


Model Configuration
-------------------

Using AutoConfig
~~~~~~~~~~~~~~~~

All models can be configured using ``AutoConfig``:

.. code-block:: python

   from tfts import AutoConfig

   # Load default configuration
   config = AutoConfig.for_model('transformer')

   # View configuration
   print(config)

   # Modify configuration
   config.hidden_size = 256
   config.num_layers = 4
   config.dropout = 0.2

   # Save configuration
   config.save_pretrained('./my_config')

   # Load configuration
   config = AutoConfig.from_pretrained('./my_config')


Common Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most models share these common parameters:

**Architecture:**
   - ``hidden_size``: Model dimension (default: 128)
   - ``num_layers``: Number of layers (default: 2-4)
   - ``dropout``: Dropout rate (default: 0.1)

**Training:**
   - ``learning_rate``: Initial learning rate
   - ``batch_size``: Training batch size
   - ``epochs``: Number of training epochs

**Input/Output:**
   - ``train_sequence_length``: Input sequence length
   - ``predict_sequence_length``: Output forecast horizon
   - ``num_features``: Number of input features


Model Comparison
----------------

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

Typical performance characteristics on standard benchmarks:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Model
     - Accuracy
     - Speed
     - Memory
     - Interpretability
   * - RNN/Seq2seq
     - ⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐
   * - TCN/WaveNet
     - ⭐⭐⭐⭐
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐
   * - Transformer
     - ⭐⭐⭐⭐
     - ⭐⭐⭐
     - ⭐⭐⭐
     - ⭐⭐⭐
   * - Informer
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐
   * - Autoformer
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐
     - ⭐⭐⭐
     - ⭐⭐⭐
   * - N-BEATS
     - ⭐⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐⭐⭐
   * - DLinear
     - ⭐⭐⭐
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐⭐


Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

Time and space complexity for different models:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Model
     - Time Complexity
     - Space Complexity
   * - RNN/LSTM
     - O(L)
     - O(H)
   * - TCN
     - O(L log L)
     - O(H)
   * - Transformer
     - O(L²)
     - O(L²)
   * - Informer
     - O(L log L)
     - O(L log L)
   * - DLinear
     - O(L)
     - O(1)

*L = sequence length, H = hidden size*


Custom Models
-------------

Creating Custom Architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create custom models by combining TFTS components:

.. code-block:: python

   import tensorflow as tf
   from tfts import AutoConfig, AutoModel
   from tfts.layers import Attention, FeedForwardNetwork

   class CustomModel(tf.keras.Model):
       def __init__(self, predict_sequence_length):
           super().__init__()
           # Use TFTS backbone
           config = AutoConfig.for_model('transformer')
           self.backbone = AutoModel.from_config(
               config,
               predict_sequence_length=predict_sequence_length
           )

           # Add custom head
           self.custom_head = tf.keras.Sequential([
               tf.keras.layers.Dense(128, activation='relu'),
               tf.keras.layers.Dense(1)
           ])

       def call(self, inputs):
           features = self.backbone(inputs)
           return self.custom_head(features)


See Also
--------

- :doc:`tutorials` - Step-by-step model training guides
- :doc:`api` - Complete API reference
- :doc:`tricks` - Tips for improving model performance
