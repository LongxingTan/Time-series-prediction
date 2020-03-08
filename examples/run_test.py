# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01

# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01


from prepare_data import PassengerData
from deepts.models.model import Model


def main():
    params={}
    x,y=PassengerData(params).get_examples(data_dir='./international-airline-passengers.csv')
    print(x.values.shape,y.shape)

    model=Model(use_model='seq2seq')
    model.test(x,y,epochs=10)


if __name__=='__main__':
    main()
