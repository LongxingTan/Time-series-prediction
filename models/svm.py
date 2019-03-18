from sklearn import svm

class Time_svm(object):
    def __init__(self):
        self.model=svm.LinearSVC() # svm.SVC()

    def train(self,x_train,y_train):
        self.model.fit(x_train,y_train)

    def loop_train(self):
        pass

    def predict(self,x):
        self.model.predict(x)

if __name__=='__main__':
    from prepare_model_input import Input_builder
    import numpy as np
    input_builder = Input_builder('../data/LSTM_data.csv')
    trainX, trainY = input_builder.create_RNN_input(time_state=5)
    trainX=np.squeeze(trainX)
    trainY=np.squeeze(trainY)

    model=Time_svm()
    model.train(trainX,trainY)