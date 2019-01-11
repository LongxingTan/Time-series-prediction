import pandas as pd



class prepare_data(object):
    def import_examples(self,data_dir):
        raise NotImplementedError

    @classmethod
    def _read_csv(cls,input_file):
        pass

    @classmethod
    def _read_excel(cls,input_file,skiprows=0):
        return pd.read_excel(input_file,skiprows=skiprows)

    @classmethod
    def _merge_by_VIN(cls,samples,failures):
        pass



class prepare_mercedes(prepare_data):
    def __init__(self):
        pass

class prepare_denza(prepare_data):
    def __init__(self):
        self.import_examples(['./production_2018.xlsx', './failure_2018.xlsx'])
        self.find_interval()
        self.find_failures()

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
        self.failures_agg=self.failures.groupby(['Repairdate'])
        print(self.samples.head(5))
        print(self.failures.head(5))
        print(self.failures_agg.head(5))

    def create_feature_samples(self):
        #create the features of samples
        pass

    def create_feature_failure_rate(self):
        #create the features of failure_rate_interval & failure_rate_cumulative
        pass

    def create_feature_auto_corr(self):
        # create the features of week_to_week auto_correlation and month_to_month_auto_correlation
        pass

    def normalize(self):
        pass




prepare_denza()