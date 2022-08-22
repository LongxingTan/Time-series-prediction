"""
This is an example of time series prediction by tfts
- multi-step prediction task
"""


from tfts import AutoConfig, AutoModel
from dataset import AutoData


def build_model(use_model):
    inputs = Input()
    config = AutoConfig(use_model)
    print(config)

    backbone = AutoModel(use_model, config)
    outputs = backbone(inputs)
    model = tf.keras.Model(inputs, outputs=outputs)

    model.compile()
    return model


def run_train():
    train_loader, valid_loader = AutoData('passenger')

    optimizer = Adam()
    loss_fn = MSE()
    model = build_model('wavenet')
    model.fit()


if __name__ == '__main__':
    run_train()

