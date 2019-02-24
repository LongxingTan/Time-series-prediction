import pandas as pd
import os
import numpy as np

class Prepare_Data(object):
    '''prepare and transfer data from /raw_data to the model input data '''
    def __init__(self):
        self.interval_dict={'daily':1,'weekly':7,'monthly':30}

    def import_examples(self,data_dir):
        raise NotImplementedError

    @classmethod
    def _read_excel(cls,input_file,skiprows=0,parse_dates=None,date_parser=None):
        return pd.read_excel(input_file,skiprows=skiprows,parse_dates=parse_dates,date_parser=date_parser)

    @classmethod
    def _read_excel_batch(cls,input_file):
        pass

    @classmethod
    def _write_csv(cls,export_data,export_name):
        export_data.to_csv('./data/'+export_name,sep=',',index=False)



class Prepare_Mercedes(Prepare_Data):
    def __init__(self,production_file,repair_file,output_file='Weibull_data.csv',interval_type='weekly'):
        super(Prepare_Mercedes, self).__init__()
        self.interval_type = interval_type
        self.output_file = output_file
        self.import_examples([production_file, repair_file])
        self.create_feature_vehicle()
        self.find_interval()
        self.find_failures()
        self.create_feature_samples()
        self.create_feature_failure_rate()

    def import_examples(self,data_dir):
        self.samples,self.failures=pd.DataFrame(),pd.DataFrame()
        for file in os.listdir(data_dir[0]):
            sample=self._read_excel(os.path.join(data_dir[0],file),
                                    parse_dates=['Production date','Initial registration date'],
                                    date_parser=lambda x: pd.datetime.strptime(str(x),"%m/%d/%Y"))
            self.samples= self.samples.append(sample, ignore_index=True)

        for file in os.listdir(data_dir[1]):
            failure=self._read_excel(os.path.join(data_dir[1],file),
                                     parse_dates=['Production date','Initial registration date','Repair date','Date of credit (sum)'],
                                     date_parser=lambda x: pd.datetime.strptime(str(x),"%m/%d/%Y"))
            self.failures=self.failures.append(failure,ignore_index=True)
        self.failures = self.failures.loc[self.failures['Total costs'] > 0, :]


    def create_feature_vehicle(self):
        self.failures['Carline']=self.failures['FIN'].astype(str).str.slice(0,3)
        self.failures['Repair month']=self.failures.loc[:,'Repair date'].apply(lambda x: x.strftime("%Y%m"))


    def find_interval(self):
        self.failures['Interval']=list(map(lambda x:x.days,self.failures['Repair date']-self.failures['Initial registration date']))
        upto_date = max(self.failures['Repair date'])
        self.samples = self._fillup_registration_date(self.samples)
        self.samples['Interval']=list(map(lambda x:x.days,upto_date-self.samples['Production date']))
        self.samples['Interval']=self.samples['Interval']+self.samples['delay']


    def find_failures(self):
        self.failures_aggby_interval = self.failures.groupby(['Interval'])['FIN'].count().reset_index(name='Failures')
        if self.interval_type=="daily":
            self.failures_aggby_calendar = self.failures.groupby(['Repair date'])['FIN'].count().reset_index(name='Failures')
        elif self.interval_type=='weekly':
            self.failures['Repair week'] = self.failures['Repair date'].apply(lambda x: x.strftime('%Y%V'))
            self.failures_aggby_calendar=self.failures.groupby(['Repair week'])['FIN'].count().reset_index(name='Failures')
        elif self.interval_type=='monthly':
            self.failures_aggby_calendar=self.failures.groupby(['Repair month'])['FIN'].count().reset_index(name='Failures')
        else:
            print("not ready for this interval type yet")
        self._write_csv(self.failures_aggby_calendar, 'LSTM_data.csv')

    def create_feature_samples(self):
        # create the features of samples
        self.samples_aggby_interval = self.samples.groupby(['Interval'])['FIN'].count().reset_index(name='Samples')
        self.samples_aggby_interval = self.samples_aggby_interval.loc[self.samples_aggby_interval['Interval'] > 0, :]

        self.samples_aggby_date = self.samples.groupby(['Initial registration date'])['FIN'].count().reset_index(name='Samples')

        self.data_aggby_interval = self.samples_aggby_interval.merge(self.failures_aggby_interval, on='Interval',
                                                                     how='left').fillna(0)
        interval_type_num = self.interval_dict[self.interval_type]
        self.data_aggby_interval['Interval'] = self.data_aggby_interval['Interval'] // interval_type_num + 1
        self.data_aggby_interval = self.data_aggby_interval.groupby(['Interval']).sum().reset_index()
        self.data_aggby_interval.sort_values(by=['Interval'], ascending=False, inplace=True)
        self.data_aggby_interval['Samples_cum'] = self.data_aggby_interval['Samples'].cumsum()
        self.data_aggby_interval.sort_values(by=['Interval'], ascending=True, inplace=True)
        self.data_aggby_interval = self.data_aggby_interval.loc[self.data_aggby_interval['Samples_cum'] > 0, :]
        # self.data_aggby_date=self.samples_aggby_date.merge(self.failures_aggby_date,left_on='retail date',right_on='Repairdate')


    def create_feature_failure_rate(self):
        # create the features of failure_rate_interval & failure_rate_cumulative
        self.data_aggby_interval['Failure_rate'] = self.data_aggby_interval['Failures'] / self.data_aggby_interval[
            'Samples_cum']
        self.data_aggby_interval['Survival_rate'] = 1 - self.data_aggby_interval['Failure_rate']
        self.data_aggby_interval['Survival_rate_cum'] = self.data_aggby_interval['Survival_rate'].cumprod()
        self.data_aggby_interval['Failure_rate_cum'] = 1 - self.data_aggby_interval['Survival_rate_cum']
        self._write_csv(self.data_aggby_interval, 'Weibull_data.csv')

    def _fillup_registration_date(self,samples):
        samples['delay']=list(map(lambda x:x.days,self.samples['Initial registration date']-self.samples['Production date']))
        Monte_cola_boolen = samples['delay'].notnull()
        Monte_cola_list = list(samples.loc[Monte_cola_boolen,'delay'])
        samples.loc[~Monte_cola_boolen,'delay'] = [np.random.choice(Monte_cola_list) for _ in range(len(samples.loc[~Monte_cola_boolen]))]
        return samples



class Prepare_Denza(Prepare_Data):
    def __init__(self,production_file,repair_file,output_file='Weibull_data.csv',interval_type='weekly',):
        super(Prepare_Denza,self).__init__()
        self.interval_type = interval_type
        self.output_file=output_file
        self.import_examples([production_file, repair_file])
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
        if self.interval_type=='daily':
            self.failures_aggby_calendar=self.failures.groupby(['Repairdate'])['Part name'].count().reset_index(name='Failures')
        elif self.interval_type=='weekly':
            self.failures['Repair week'] = self.failures['Repairdate'].apply(lambda x: x.strftime('%Y%V'))
            self.failures_aggby_calendar=self.failures.groupby(['Repair week'])['Part name'].count().reset_index(name='Failures')
        elif self.interval_type=='monthly':
            self.failures['Repair month'] = self.failures['Repairdate'].apply(lambda x: x.strftime('%Y%m'))
            self.failures_aggby_calendar=self.failures.groupby(['Repair month'])['Part name'].count().reset_index(name='Failures')
        self.failures_aggby_interval=self.failures.groupby(['Interval'])['VIN'].count().reset_index(name='Failures')



    def create_feature_samples(self):
        #create the features of samples
        self.samples_aggby_date = self.samples.groupby(['retail date'])['Model year'].count().reset_index(name='Samples')
        self.samples_aggby_interval = self.samples.groupby(['Interval'])['Model year'].count().reset_index(name='Samples')
        self.samples_aggby_interval = self.samples_aggby_interval.loc[self.samples_aggby_interval['Interval'] > 0, :]

        interval_full=pd.DataFrame({'Interval':range(int(0.95*max(self.samples_aggby_interval['Interval'])))})
        self.samples_aggby_interval=self.samples_aggby_interval.merge(interval_full,on='Interval',how='right').fillna(0)

        self.data_aggby_interval = self.samples_aggby_interval.merge(self.failures_aggby_interval, on='Interval',
                                                                     how='left').fillna(0)
        interval_type_num=self.interval_dict[self.interval_type]
        self.data_aggby_interval['Interval'] = self.data_aggby_interval['Interval'] //interval_type_num + 1
        self.data_aggby_interval=self.data_aggby_interval.groupby(['Interval']).sum().reset_index()
        self.data_aggby_interval.sort_values(by=['Interval'],ascending=False,inplace=True)
        self.data_aggby_interval['Samples_cum'] = self.data_aggby_interval['Samples'].cumsum()
        self.data_aggby_interval.sort_values(by=['Interval'], ascending=True, inplace=True)
        self.data_aggby_interval=self.data_aggby_interval.loc[self.data_aggby_interval['Samples_cum'] > 0, :]
        #self.data_aggby_date=self.samples_aggby_date.merge(self.failures_aggby_date,left_on='retail date',right_on='Repairdate')


    def create_feature_failure_rate(self):
        #create the features of failure_rate_interval & failure_rate_cumulative
        self.data_aggby_interval['Failure_rate']=self.data_aggby_interval['Failures']/self.data_aggby_interval['Samples_cum']
        self.data_aggby_interval['Survival_rate']=1-self.data_aggby_interval['Failure_rate']
        self.data_aggby_interval['Survival_rate_cum']=self.data_aggby_interval['Survival_rate'].cumprod()
        self.data_aggby_interval['Failure_rate_cum']=1-self.data_aggby_interval['Survival_rate_cum']
        self._write_csv(self.data_aggby_interval, self.output_file)


    def create_feature_auto_corr(self):
        # create the features of week_to_week auto_correlation and month_to_month_auto_correlation
        """
                       http://stackoverflow.com/q/14297012/190597
                       http://en.wikipedia.org/wiki/Autocorrelation#Estimation
                       """
        n = len(x)
        variance = x.var()
        x = x - x.mean()
        r = np.correlate(x, x, mode='full')[-n:]
        assert np.allclose(r, np.array([(x[:n - k] * x[-(n - k):]).sum() for k in range(n)]))
        result = r / (variance * (np.arange(n, 0, -1)))

        pass


    def create_feature_vehicle(self):
        pass

    def create_batch_features_weibull(self):
        pass

    def create_batch_features_dl(self):
        pass

    def normalize(self):
        pass

    def power_transform(self):
        # https://machinelearningmastery.com/machine-learning-data-transforms-for-time-series-forecasting/
        pass

    def exponoential_smooth(self,series,alpha):
        result=[series[0]]
        for n in range(1,len(series)):
            result.append(alpha*series[n]+(1-alpha)*result[n-1])
        return result


if __name__=="__main__":
    mercedes=Prepare_Mercedes(interval_type='monthly')

    '''
    denza = Prepare_Denza(interval_type='weekly')
    from models.weibull import *
    weibull = Weibull_model()
    weibull._weibull_density_plot(denza.data_aggby_interval['Interval'], denza.data_aggby_interval['Failure_rate'])
    '''

