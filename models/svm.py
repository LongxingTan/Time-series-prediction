from sklearn import svm

class Time_SVM(object):
    def __init__(self):
        self.model=svm.LinearSVC() # svm.SVC()

    def train(self,train):
        self.model.fit(x_train,y_train)

    def loop_train(self):
        pass

    def predict(self,x):
        self.model.predict(x)

