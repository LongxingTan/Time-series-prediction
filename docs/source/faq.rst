Frequently Asked Questions (FAQ)
==================================

This page answers common questions about TFTS. If you don't find your answer here, please ask in `GitHub Discussions <https://github.com/LongxingTan/Time-series-prediction/discussions>`_.


General Questions
-----------------

What is TFTS?
~~~~~~~~~~~~~

TFTS (TensorFlow Time Series) is a comprehensive Python library providing state-of-the-art deep learning models for time series analysis. It offers 20+ pre-implemented models, a unified API, and production-ready features for forecasting, classification, and anomaly detection.


What can I use TFTS for?
~~~~~~~~~~~~~~~~~~~~~~~~~

TFTS supports multiple time series tasks:

- **Forecasting:** Single/multi-step, univariate/multivariate predictions
- **Probabilistic Forecasting:** Uncertainty quantification with confidence intervals
- **Classification:** Time series classification tasks
- **Anomaly Detection:** Identifying outliers and anomalies
- **Segmentation:** Change point detection


How does TFTS compare to other libraries?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**vs. Prophet:**
   - TFTS: Deep learning models, flexible architecture, GPU support
   - Prophet: Statistical models, interpretable, good for business forecasting

**vs. NeuralProphet:**
   - TFTS: 20+ models, TensorFlow-based, production-ready
   - NeuralProphet: PyTorch-based, Prophet-like interface

**vs. GluonTS:**
   - TFTS: TensorFlow ecosystem, Keras integration
   - GluonTS: MXNet/PyTorch, probabilistic focus

**vs. Darts:**
   - TFTS: Specialized for deep learning, extensive model selection
   - Darts: Both classical and DL models, comprehensive toolkit


Installation & Setup
--------------------

How do I install TFTS?
~~~~~~~~~~~~~~~~~~~~~~

The easiest way:

.. code-block:: bash

   pip install tfts

For development:

.. code-block:: bash

   git clone https://github.com/LongxingTan/Time-series-prediction.git
   cd Time-series-prediction
   pip install -e .

See :doc:`installation` for detailed instructions.


Which TensorFlow version should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Recommended:** TensorFlow >= 2.4

- **TensorFlow 2.4-2.8:** Stable, well-tested
- **TensorFlow 2.9+:** Latest features, best performance
- **TensorFlow 2.15+:** Keras 3.0 support (experimental)

Check compatibility:

.. code-block:: python

   import tensorflow as tf
   print(f"TensorFlow version: {tf.__version__}")


Do I need a GPU?
~~~~~~~~~~~~~~~~

**No, but recommended.**

- **CPU:** Works fine for small datasets and prototyping
- **GPU:** Significantly faster for large datasets and complex models
- **TPU:** Best for very large-scale training

GPU speedup is typically 10-50x faster than CPU.


How do I enable GPU support?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install CUDA toolkit and cuDNN
2. Install TensorFlow with GPU support:

.. code-block:: bash

   pip install tensorflow[and-cuda]

3. Verify:

.. code-block:: python

   import tensorflow as tf
   print("GPUs Available:", len(tf.config.list_physical_devices('GPU')))


Data & Models
-------------

What data format does TFTS accept?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TFTS accepts multiple formats:

1. **NumPy arrays:**

.. code-block:: python

   x_train = np.array([...])  # Shape: (samples, timesteps, features)
   y_train = np.array([...])  # Shape: (samples, pred_length, 1)

2. **Pandas DataFrames (via TimeSeriesSequence):**

.. code-block:: python

   from tfts.data import TimeSeriesSequence

   data_loader = TimeSeriesSequence(
       data=df,
       time_idx='timestamp',
       target_column='target',
       train_sequence_length=24,
       predict_sequence_length=8
   )

3. **TensorFlow datasets:**

.. code-block:: python

   dataset = tf.data.Dataset.from_tensor_slices((x, y))


Which model should I choose?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Quick Guide:**

- **Starting out:** ``seq2seq`` or ``dlinear`` (simple, fast)
- **Best accuracy:** ``informer``, ``autoformer``, or ``transformer``
- **Long sequences:** ``informer`` or ``patch_tst``
- **Interpretability:** ``nbeats`` or ``dlinear``
- **Uncertainty:** ``deep_ar`` or ``diffusion``
- **Speed:** ``tcn`` or ``dlinear``

See :doc:`models` for detailed comparisons.


How much data do I need?
~~~~~~~~~~~~~~~~~~~~~~~~~

**Minimum:**
   - Simple models (RNN, TCN): 500-1000 samples
   - Transformer models: 2000-5000 samples
   - Complex models (Informer, Autoformer): 5000+ samples

**Recommended:**
   - 10,000+ samples for robust training
   - More data → better generalization
   - Data augmentation can help with small datasets


Can I use TFTS for multivariate time series?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Yes!** TFTS fully supports multivariate forecasting:

.. code-block:: python

   # Multiple input features, single target
   x = np.random.randn(1000, 24, 10)  # 10 features
   y = np.random.randn(1000, 8, 1)    # 1 target

   # Multiple targets
   y = np.random.randn(1000, 8, 3)    # 3 targets

Models like ``itransformer`` and ``tft`` are specifically designed for multivariate data.


Training & Performance
----------------------

How long does training take?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Typical training times (on single GPU):**

- Simple models (RNN, DLinear): Minutes
- TCN/WaveNet: 10-30 minutes
- Transformer: 30-60 minutes
- Informer/Autoformer: 1-2 hours

**Factors:**
   - Dataset size
   - Sequence length
   - Model complexity
   - Batch size
   - Hardware


My model isn't learning. What should I do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Checklist:**

1. **Check data:**

.. code-block:: python

   # Verify shapes
   print(f"X shape: {x_train.shape}")
   print(f"Y shape: {y_train.shape}")

   # Check for NaN/inf
   assert not np.isnan(x_train).any()
   assert not np.isinf(x_train).any()

2. **Normalize data:**

.. code-block:: python

   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   x_train_scaled = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1]))
   x_train_scaled = x_train_scaled.reshape(x_train.shape)

3. **Try simpler model:**

.. code-block:: python

   # Start with RNN or DLinear
   config = AutoConfig.for_model('rnn')
   model = AutoModel.from_config(config, predict_sequence_length=8)

4. **Adjust learning rate:**

.. code-block:: python

   # Try lower learning rate
   optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

5. **Check loss function:**

.. code-block:: python

   # For forecasting, use MSE or MAE
   loss = tf.keras.losses.MeanSquaredError()


How can I improve model performance?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the comprehensive :doc:`tricks` guide. Quick tips:

1. **Feature engineering:** Add datetime, lag, and rolling features
2. **Hyperparameter tuning:** Adjust hidden_size, num_layers, dropout
3. **Ensemble models:** Combine multiple models
4. **Data augmentation:** Add noise, jittering
5. **Longer training:** More epochs with early stopping


How do I prevent overfitting?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Strategies:**

1. **Dropout:**

.. code-block:: python

   config.dropout = 0.2
   config.attention_probs_dropout_prob = 0.1

2. **Early stopping:**

.. code-block:: python

   early_stop = tf.keras.callbacks.EarlyStopping(
       monitor='val_loss',
       patience=10,
       restore_best_weights=True
   )
   trainer.train(..., callbacks=[early_stop])

3. **L2 regularization:**

.. code-block:: python

   config.weight_decay = 0.01

4. **Reduce model complexity:**
   - Decrease hidden_size
   - Reduce num_layers
   - Use simpler model


Can I use pretrained models?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Yes, for transfer learning:**

.. code-block:: python

   # Save model
   model.save_pretrained('./my_model')

   # Load pretrained weights
   new_model = AutoModel.from_pretrained('./my_model')

   # Fine-tune on new data
   trainer = KerasTrainer(new_model)
   trainer.train(new_data, epochs=10)


Production & Deployment
-----------------------

How do I deploy TFTS models?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: TensorFlow Serving**

.. code-block:: python

   # Save model
   model.save('./saved_model')

.. code-block:: bash

   # Serve with TensorFlow Serving
   tensorflow_model_server --model_base_path=/path/to/saved_model

**Option 2: ONNX export**

.. code-block:: python

   import tf2onnx

   # Convert to ONNX
   model_proto, _ = tf2onnx.convert.from_keras(model)

**Option 3: Custom API**

.. code-block:: python

   from flask import Flask, request
   import joblib

   app = Flask(__name__)
   model = joblib.load('model.pkl')

   @app.route('/predict', methods=['POST'])
   def predict():
       data = request.json
       prediction = model.predict(data)
       return {'prediction': prediction.tolist()}


How do I handle missing values in production?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Strategies:**

1. **Forward fill:**

.. code-block:: python

   df = df.fillna(method='ffill')

2. **Interpolation:**

.. code-block:: python

   df = df.interpolate(method='linear')

3. **Model-based imputation:**

.. code-block:: python

   from sklearn.impute import KNNImputer

   imputer = KNNImputer(n_neighbors=5)
   df_imputed = imputer.fit_transform(df)


Can I use TFTS with streaming data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Yes!** Process data in batches:

.. code-block:: python

   import queue

   data_queue = queue.Queue()

   def process_stream():
       while True:
           # Get batch from stream
           batch = data_queue.get(timeout=1)

           # Generate features
           batch_processed = process_features(batch)

           # Predict
           prediction = model.predict(batch_processed)

           # Send results
           send_prediction(prediction)


How do I monitor model performance in production?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Monitoring checklist:**

1. **Prediction latency:**

.. code-block:: python

   import time

   start = time.time()
   prediction = model.predict(x)
   latency = time.time() - start
   print(f"Latency: {latency:.3f}s")

2. **Prediction accuracy:**
   - Track MAE/MSE on incoming data
   - Compare with ground truth when available
   - Set up alerts for degradation

3. **Data drift:**
   - Monitor input feature distributions
   - Check for out-of-distribution samples

4. **Model updates:**
   - Retrain periodically with new data
   - A/B test new models before deployment


Advanced Topics
---------------

Can I customize model architectures?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Yes!** Several approaches:

1. **Modify config:**

.. code-block:: python

   config = AutoConfig.for_model('transformer')
   config.hidden_size = 256
   config.num_layers = 6
   config.num_attention_heads = 8

2. **Custom head:**

.. code-block:: python

   model = AutoModel.from_config(config, predict_sequence_length=24)
   model.project = tf.keras.Sequential([
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(1)
   ])

3. **Fully custom model:**

.. code-block:: python

   class CustomModel(tf.keras.Model):
       def __init__(self):
           super().__init__()
           self.backbone = AutoModel.from_config(config)
           self.custom_layers = ...

       def call(self, inputs):
           features = self.backbone(inputs)
           return self.custom_layers(features)


How do I implement custom loss functions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CustomLoss(tf.keras.losses.Loss):
       def call(self, y_true, y_pred):
           # Your custom loss logic
           mse = tf.reduce_mean(tf.square(y_true - y_pred))
           mae = tf.reduce_mean(tf.abs(y_true - y_pred))
           return mse + 0.1 * mae

   # Use in training
   trainer = KerasTrainer(model, loss_fn=CustomLoss())


Can I use attention weights for interpretability?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Yes, for models with attention:**

.. code-block:: python

   # Get model with attention outputs
   model = AutoModel.from_config(config, output_attention=True)

   # Make prediction
   outputs = model(x, output_attention=True)
   predictions = outputs['predictions']
   attention_weights = outputs['attention_weights']

   # Visualize attention
   import matplotlib.pyplot as plt
   plt.imshow(attention_weights[0], cmap='hot')
   plt.colorbar()
   plt.show()


Troubleshooting
---------------

I'm getting OOM (Out of Memory) errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solutions:**

1. **Reduce batch size:**

.. code-block:: python

   batch_size = 16  # or smaller

2. **Use gradient accumulation:**

.. code-block:: python

   # Accumulate gradients over multiple batches
   accumulation_steps = 4
   effective_batch_size = batch_size * accumulation_steps

3. **Mixed precision training:**

.. code-block:: python

   from tensorflow.keras.mixed_precision import set_global_policy

   set_global_policy('mixed_float16')

4. **Reduce model size:**
   - Decrease hidden_size
   - Reduce num_layers
   - Use model distillation


Training is very slow
~~~~~~~~~~~~~~~~~~~~~~

**Optimization tips:**

1. **Use GPU:**

.. code-block:: python

   with tf.device('/GPU:0'):
       trainer.train(...)

2. **Increase batch size:**
   - Larger batches = fewer iterations
   - Limited by memory

3. **Use tf.data pipeline:**

.. code-block:: python

   dataset = tf.data.Dataset.from_tensor_slices((x, y))
   dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

4. **Mixed precision:**

.. code-block:: python

   set_global_policy('mixed_float16')


Model predictions are all the same
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Common causes:**

1. **Dead neurons:** Try lower learning rate or different activation
2. **Poor initialization:** Check weight initialization
3. **Wrong normalization:** Verify data preprocessing
4. **Too much regularization:** Reduce dropout/weight decay


Getting Help
------------

Where can I get help?
~~~~~~~~~~~~~~~~~~~~~

**Community:**
   - 💬 `GitHub Discussions <https://github.com/LongxingTan/Time-series-prediction/discussions>`_ - Ask questions
   - 🐛 `GitHub Issues <https://github.com/LongxingTan/Time-series-prediction/issues>`_ - Report bugs
   - 📖 `Documentation <https://time-series-prediction.readthedocs.io>`_ - Read the docs


How do I report a bug?
~~~~~~~~~~~~~~~~~~~~~~~

**When reporting bugs, please include:**

1. TFTS version: ``import tfts; print(tfts.__version__)``
2. TensorFlow version: ``import tensorflow as tf; print(tf.__version__)``
3. Python version: ``import sys; print(sys.version)``
4. Operating system
5. Full error traceback
6. Minimal reproducible example


How can I contribute?
~~~~~~~~~~~~~~~~~~~~~~

We welcome contributions! See `CONTRIBUTING.md <https://github.com/LongxingTan/Time-series-prediction/blob/master/CONTRIBUTING.md>`_ for guidelines.

**Ways to contribute:**
   - Report bugs and suggest features
   - Improve documentation
   - Add new models or features
   - Fix bugs
   - Add tests
   - Share examples and tutorials


See Also
--------

- :doc:`installation` - Installation guide
- :doc:`tutorials` - Step-by-step tutorials
- :doc:`models` - Model documentation
- :doc:`tricks` - Performance tips
- :doc:`api` - API reference
