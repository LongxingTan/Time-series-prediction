"""Demo to tune the model parameters by Autotune"""

import optuna

import tfts
from tfts import AutoConfig, AutoModel, AutoTuner, KerasTrainer
