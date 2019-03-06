# feature engineering refer to:  https://github.com/Arturus/kaggle-web-traffic
import os
import numpy as np
import pandas as pd
from prepare_data import Prepare_Data

# feature based on the calendar time
class Prepare_Mercedes_calendar(Prepare_Data):
    def __init__(self, failure_file,interval_type='weekly',failure_type='fault location'):
        super(Prepare_Mercedes_calendar,self).__init__()
        self.interval_type=interval_type
        self.failure_type=failure_type
        failures=self.import_examples(failure_file)
        self.failures_aggby_calendar=self.aggregate_failures(failures)
        print('shape',self.failures_aggby_calendar.shape)
        start_ids,stop_ids=self.choose_start_stop(self.failures_aggby_calendar.values)

        raw_year_autocorr=self.batch_autocorr(self.failures_aggby_calendar.values,int(365.25//7),start_ids,stop_ids,1.5)
        year_unknown_pct=np.sum(np.isnan(raw_year_autocorr))/len(raw_year_autocorr)

        raw_quarter_autocorr=self.batch_autocorr(self.failures_aggby_calendar.values,int(round(365.25/4//7)),start_ids,stop_ids,2)
        quarter_unknown_pct=np.sum(np.isnan(raw_quarter_autocorr))/len(raw_quarter_autocorr)

        print("Percent of undefined autocorr = yearly:%.3f, quarterly:%.3f" % (year_unknown_pct, quarter_unknown_pct))


        year_autocorr = self.normalize(np.nan_to_num(raw_year_autocorr))
        quarter_autocorr = self.normalize(np.nan_to_num(raw_quarter_autocorr)) # => [batch_size]

        auto_corr_features = np.stack([quarter_autocorr, year_autocorr])

        print(self.failures_aggby_calendar.columns)


        self.lagged_ix=self.lag_indexes(self.failures_aggby_calendar,offsets=[1,3,5]) #13,26,52 weeks

        print(self.lagged_ix.shape)


    def import_examples(self,data_dir):
        failures=pd.DataFrame()
        for file in os.listdir(data_dir):
            failure=self._read_excel(os.path.join(data_dir,file),
                                     parse_dates=['Production date','Initial registration date','Repair date','Date of credit (sum)'],
                                     date_parser=lambda x: pd.datetime.strptime(str(x),"%m/%d/%Y"))
            failures=failures.append(failure,ignore_index=True)
        failures = failures.loc[failures['Total costs'] > 0, :]
        failures['Failure location']=failures['Unnamed: 11']
        failures['Failure type']=failures['Unnamed: 13']
        return failures

    def aggregate_failures(self,failures):
        if self.failure_type=='fault location':
            failure_type_agg='Failure location'
        elif self.failure_type=='damage code':
            failures['Failure']=failures['Failure location']+' '+failures['Failure type']
            failure_type_agg='Failure'
        else: raise ValueError ("Unsupported failure type %s" % self.failure_type)

        if self.interval_type=="daily":
            failures_aggby_calendar = failures.groupby(['Repair date',failure_type_agg])['FIN'].count().reset_index(name='Failures')
        elif self.interval_type=='weekly':
            failures['Repair week'] = failures['Repair date'].apply(lambda x: x.strftime('%Y%V'))
            failures_aggby_calendar=failures.groupby(['Repair week',failure_type_agg]).size().reset_index(name='Failures')
            failures_aggby_calendar = failures_aggby_calendar.pivot(index='Failure location', columns='Repair week', values='Failures')
            failures_aggby_calendar=np.log1p(failures_aggby_calendar.fillna(0))
        elif self.interval_type=='monthly':
            failures['Repair month'] = failures.loc[:, 'Repair date'].apply(lambda x: x.strftime("%Y%m"))
            failures_aggby_calendar=failures.groupby(['Repair month',failure_type_agg])['FIN'].count().reset_index(name='Failures')
        else: raise ValueError("Unsupported interval type %s" % self.interval_type)
        return failures_aggby_calendar

    def choose_start_stop(self,input:np.ndarray):
        n_issue,n_time=input.shape
        start_idx=np.full(n_issue,-1,dtype=np.int32)
        end_idx=np.full(n_issue,-1,dtype=np.int32)
        for issue in range(n_issue):
            for time in range(n_time):
                if not np.isnan(input[issue,time]) and input[issue,time]>0:
                    start_idx[issue]=time
                    break
            for time in range(n_time-1,-1,-1):
                if not np.isnan(input[issue,time]) and input[issue,time]>0:
                    end_idx[issue]=time
                    break
        return start_idx,end_idx

    def mask_short_series(self,data,start_idx,end_idx,threshold):
        issue_mask=(end_idx-start_idx)/data.shape[1]<threshold
        print("Masked %d issues from %d" % (issue_mask.sum(),len(data)))
        inv_mask=~issue_mask
        data=data[inv_mask]
        return data


    def batch_autocorr(self,data,lag,starts,ends,threshold,backoffdet=0):
        # auto correlation as feature
        n_series,n_time=data.shape
        max_end=n_time-backoffdet
        corr=np.empty(n_series,dtype=np.float32)
        support=np.empty(n_series,dtype=np.float32)
        for i in range(n_series):
            series=data[i]
            end=min(ends[i],max_end)
            real_len=end-starts[i]
            support[i]=real_len/lag
            if support[i]>threshold:
                series=series[starts[i]:end]
                c_365=self._single_autocorr(series,lag)
                c_364=self._single_autocorr(series,lag-1)
                c_366=self._single_autocorr(series,lag+1)
                corr[i]=0.5*c_365+0.25*c_364+0.25*c_366
            else:
                corr[i]=np.NaN
        return corr

    def _single_autocorr(self,series,lag):
        s1=series[lag:]
        s2=series[:-lag]
        s1_nor=s1-np.mean(s1)
        s2_nor=s2-np.mean(s2)
        divider=np.sqrt(np.sum(s1_nor*s1_nor))*np.sqrt(np.sum(s2_nor*s2_nor))
        return np.sum(s1_nor*s2_nor)/divider if divider!=0 else 0

    def lag_indexes(self,data,offsets):
        # for AutoRegressive
        dr=data.columns
        base_index = pd.Series(np.arange(0, len(dr)), index=dr)
        new_indexs=[]
        for offset in offsets:
            new_index=np.roll(base_index,offset)
            new_index[:offset]=-1
            new_indexs.append(new_index)
        new_indexs=np.array(new_indexs)
        return new_indexs

    def create_vehicle_features(self,data):
        # carline, long or short version, as feature
        pass

    def create_time_features(self,data):
        # month,weekday as feature
        features_days = pd.date_range(data_start, data_end)
        week_period = 7 / (2 * np.pi)
        dow_norm = features_days.dayofweek.values / week_period
        dow = np.stack([np.cos(dow_norm), np.sin(dow_norm)], axis=-1)


if __name__ == '__main__':
    mercedes=Prepare_Mercedes_calendar(failure_file = './raw_data/failures')
    data=mercedes.failures_aggby_calendar.values[:,:10]

    lagged_ix = mercedes.lagged_ix[:,:10]
    lag_mask=lagged_ix<0
    cropped_lags=np.maximum(lagged_ix,0)
    lagged_frequency=np.take(data,cropped_lags)
    lag_zeros=np.zeros_like(lagged_frequency)
    lagged_frequency=np.where(np.logical_or(lag_mask,np.isnan(lagged_frequency)),lag_zeros,lagged_frequency)

    data_df=pd.DataFrame(data)
    data_df.to_csv("data.csv")
    new_df=pd.DataFrame(lagged_frequency)
    new_df.to_csv('new.csv')


