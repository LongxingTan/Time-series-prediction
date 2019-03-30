

def moving_avg_time(dataframe, window_size):
    # dataframe is Sereis with index
    # window_size =["30T","60T","M".....]
    first_column=dataframe.columns[0]
    dataframe.set_index(first_column,inplace=True)
    out = dataframe.resample(window_size).mean()
    return out

def moving_avg(dataframe,window_size):
    dataframe['index']=range(len(dataframe))
    dataframe['temp']=dataframe['index'].apply(lambda x: x//window_size)
    dataframe_new=dataframe.groupby(['temp']).mean().reset_index()
    dataframe_new.drop(columns=['temp','index'],inplace=True)
    return dataframe_new
