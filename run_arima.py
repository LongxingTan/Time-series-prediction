from models.ARIMA import Time_ARIMA
from prepare_model_input import Input_builder


class Config:
    n_states=5
    n_features=1

    n_layers = 1
    hidden_size=[128]
    learning_rate=10e-3
    n_epochs=15
    batch_size=1


def run_pred():
    config=Config
    model=Time_ARIMA(config)

    input_builder = Input_builder('./data/LSTM_data.csv')
    train = input_builder.create_arima_input()
    model.train(train)

if __name__=="__main__":
    run_pred()