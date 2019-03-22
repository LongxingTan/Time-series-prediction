import os
import logging
import pandas as pd
import numpy as np
from config import params
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# load Mercedes data
class Load_Mercedes(object):
    def __init__(self,production_file,repair_file,output_date_type='calendar',interval_type='weekly'):
        # process the data into operating time or time series calendar time
        self.output_date_type=output_date_type
        self.interval_type=interval_type
        self.interval_dict = {'daily': 1, 'weekly': 7, 'monthly': 30}
        self.import_examples([production_file, repair_file])
        self.aggregate_failures()
        if output_date_type!='calendar':
            self.aggregate_samples()
            self.create_feature_failure_rate()
            self._write_csv(self.data_aggby_interval, './data/interval_data.csv')
        self._write_csv(self.failures_aggby_calendar,params['calendar_data'])


    def import_examples(self, data_dir):
        self.samples, self.failures = pd.DataFrame(), pd.DataFrame()
        for file in os.listdir(data_dir[0]):
            sample = self._read_excel(os.path.join(data_dir[0], file),
                                      parse_dates=['Production date', 'Initial registration date'],
                                      date_parser=lambda x: pd.datetime.strptime(str(x), "%m/%d/%Y"))
            self.samples = self.samples.append(sample, ignore_index=True)
        logging.info("Load {} units samples ".format(len(self.samples)))

        for file in os.listdir(data_dir[1]):
            failure = self._read_excel(os.path.join(data_dir[1], file),
                                       parse_dates=['Production date', 'Initial registration date', 'Repair date',
                                                    'Date of credit (sum)'],
                                       date_parser=lambda x: pd.datetime.strptime(str(x), "%m/%d/%Y"))
            self.failures = self.failures.append(failure, ignore_index=True)
        self.failures = self.failures.loc[self.failures['Total costs'] > 0, :]
        logging.info("Load {} units valid failures".format(len(self.failures)))


    def aggregate_failures(self):
        if self.output_date_type!='calendar':
            self.find_interval()
            self.failures['Carline'] = self.failures['FIN'].astype(str).str.slice(0, 3)
            self.failures['Repair month'] = self.failures.loc[:, 'Repair date'].apply(lambda x: x.strftime("%Y%m"))

            self.failures_aggby_interval = self.failures.groupby(['Interval'])['FIN'].count().reset_index(
                name='Failures')

        if self.interval_type == "daily":
            self.failures_aggby_calendar = self.failures.groupby(['Repair date'])['FIN'].count().reset_index(
                name='Failures')
        elif self.interval_type == 'weekly':
            self.failures['Repair week'] = self.failures['Repair date'].apply(lambda x: x.strftime('%Y%V'))
            self.failures_aggby_calendar = self.failures.groupby(['Repair week'])['FIN'].count().reset_index(
                name='Failures')
        elif self.interval_type == 'monthly':
            self.failures_aggby_calendar = self.failures.groupby(['Repair month'])['FIN'].count().reset_index(
                name='Failures')
        else:
            print("not ready for this interval type yet")


    def find_interval(self):
        self.failures['Interval']=list(map(lambda x:x.days,self.failures['Repair date']-self.failures['Initial registration date']))
        upto_date = max(self.failures['Repair date'])
        self.samples = self._fillup_registration_date(self.samples)
        self.samples['Interval']=list(map(lambda x:x.days,upto_date-self.samples['Production date']))
        self.samples['Interval']=self.samples['Interval']+self.samples['delay']

    def aggregate_samples(self):
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

    def _fillup_registration_date(self,samples):
        samples['delay']=list(map(lambda x:x.days,self.samples['Initial registration date']-self.samples['Production date']))
        Monte_cola_boolen = samples['delay'].notnull()
        Monte_cola_list = list(samples.loc[Monte_cola_boolen,'delay'])
        samples.loc[~Monte_cola_boolen,'delay'] = [np.random.choice(Monte_cola_list) for _ in range(len(samples.loc[~Monte_cola_boolen]))]
        return samples

    def _read_excel(self, input_file, skiprows=0, parse_dates=None, date_parser=None):
        return pd.read_excel(input_file, skiprows=skiprows, parse_dates=parse_dates, date_parser=date_parser)

    def _write_csv(self, export_data, export_name):
        export_data.to_csv(export_name, sep=',', index=False)


# load the EMG data
def load_EMG_data():
    pass


# load the jet engine data
