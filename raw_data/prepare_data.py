import pandas as pd

class Prepare_Data(object):
    def __init__(self):
        self.interval_dict={'daily':1,'weekly':7,'monthly':30}

    def import_examples(self,data_dir):
        raise NotImplementedError

    @classmethod
    def _read_csv(cls,input_file):
        pass

    @classmethod
    def _read_excel(cls,input_file,skiprows=0):
        return pd.read_excel(input_file,skiprows=skiprows)


class Prepare_Mercedes(Prepare_Data):
    def __init__(self):
        super(Prepare_Mercedes, self).__init__()
        pass


class Prepare_Denza(Prepare_Data):
    def __init__(self,interval_type='daily'):
        super(Prepare_Denza,self).__init__()
        self.interval_type=interval_type
        self.import_examples(['./production_2018.xlsx', './failure_2018.xlsx'])
        self.find_interval()
        self.find_failures()
        self.create_feature_samples()
        self.create_feature_failure_rate()

    def import_examples(self,data_dir):
        self.samples=self._read_excel(data_dir[0],skiprows=1)
        self.failures=self._read_excel(data_dir[1],skiprows=1)
        self.samples['retail date']=pd.to_datetime(self.samples['retail date'],format='%Y-%m-%d',errors='ignore')
        self.failures.columns=self.failures.columns.str.replace('\n','')
        self.failures['Repairdate']=pd.to_datetime(self.failures['Repairdate'],format='%Y-%m-%d',errors='ignore')
        self.failures['Deliverydate']=pd.to_datetime(self.failures['handover date-Update(customer)(retail by 31/12/2018)'],
                                                     format='%Y-%m-%d',errors='ignore')

    def find_interval(self):
        #make the known time series
        self.failures['Interval']=list(map(lambda x: int(x.days),self.failures['Repairdate']-self.failures['Deliverydate']))

        upto_date=max(self.failures['Repairdate'])
        self.samples['Interval']=list(map(lambda x:int(x.days),upto_date-self.samples['retail date']))


    def find_failures(self):
        #make the predicted failures & failures_diff
        self.failures_aggby_date=self.failures.groupby(['Repairdate'])['Part name'].count()
        self.failures_aggby_interval=self.failures.groupby(['Interval'])['Part name'].count().reset_index(name='Failures')


    def create_feature_samples(self):
        #create the features of samples
        self.samples_aggby_date = self.samples.groupby(['retail date'])['Model year'].count()
        self.samples_aggby_interval = self.samples.groupby(['Interval'])['Model year'].count().reset_index(
            name='Samples')
        self.samples_aggby_interval = self.samples_aggby_interval.loc[self.samples_aggby_interval['Interval'] > 0, :]

        self.data_aggby_interval = self.samples_aggby_interval.merge(self.failures_aggby_interval, on='Interval',
                                                                     how='left').fillna(0)
        interval_type_num=self.interval_dict[self.interval_type]
        self.data_aggby_interval['Interval'] = self.data_aggby_interval['Interval'] //interval_type_num + 1
        self.data_aggby_interval=self.data_aggby_interval.groupby(['Interval']).sum().reset_index()
        #print(self.data_aggby_interval)
        self.data_aggby_interval['Samples_cum'] = self.data_aggby_interval['Samples'].cumsum()
        #self.data_aggby_date=self.samples_aggby_date.merge(self.failures_aggby_date,left_on='retail date',right_on='Repairdate')




    def create_feature_failure_rate(self):
        #create the features of failure_rate_interval & failure_rate_cumulative
        self.data_aggby_interval['Failure_rate']=self.data_aggby_interval['Failures']/self.data_aggby_interval['Samples_cum']
        self.data_aggby_interval['Survival_rate']=1-self.data_aggby_interval['Failure_rate']
        self.data_aggby_interval['Survival_rate_cum']=self.data_aggby_interval['Survival_rate'].cumprod()
        self.data_aggby_interval['Failure_rate_cum']=1-self.data_aggby_interval['Survival_rate_cum']



        self.data_aggby_interval.to_csv('../data/Weibull_data.csv',sep=',',index=False)

    def create_feature_auto_corr(self):
        # create the features of week_to_week auto_correlation and month_to_month_auto_correlation
        pass

    def normalize(self):
        pass


if __name__=="__main__":
    denza = Prepare_Denza()
    from models.weibull import *
    weibull = Weibull()
    weibull._weibull_density_plot(denza.data_aggby_interval['Interval'], denza.data_aggby_interval['Failure_rate'])

