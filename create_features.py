import logging
import pandas as pd
import numpy as np

epsilon=10e-7
class Create_features(object):
    def __init__(self):
        pass

    def __call__(self,data, *args, **kwargs):
        start_ids, stop_ids = self.choose_start_stop(data.values)

        history_as_feature = self.create_history_as_feature(data.values)
        median_as_feature = self.create_median_as_feature(data.values)
        time_style_feature = self.create_time_features(data)
        lagged_feature = self.create_auto_regression_feature(data=data, offsets=[4, 8])
        auto_corr_features = self.create_auto_correlation_feature(data.values, start_ids,stop_ids)
        self.features = np.concatenate(
            [history_as_feature, median_as_feature, time_style_feature, lagged_feature, auto_corr_features], axis=-1)

    def create_history_as_feature(self,data):
        # batch_size * n_time => batch_size * n_time *1
        mean=np.mean(data,axis=-1,keepdims=True)
        std=np.std(data,axis=-1,keepdims=True)
        history_as_feature=(data-mean)/(std+epsilon)
        history_as_feature=np.expand_dims(history_as_feature,-1)
        return history_as_feature

    def create_auto_regression_feature(self,data,offsets):
        # features for auto regression,  batch_size *n_time=> batch_size* n_time * len(offsets)
        lagged_ix = self.lag_indexes(data, offsets=offsets)  # 13,26,52 weeks
        lag_mask = lagged_ix < 0
        cropped_lags = np.maximum(lagged_ix, 0)

        lagged_feature = np.take(data.values, cropped_lags, axis=-1)
        lag_zeros = np.zeros_like(lagged_feature)
        lagged_feature = np.where(np.logical_or(lag_mask, np.isnan(lagged_feature)), lag_zeros, lagged_feature)
        return lagged_feature

    def create_vehicle_features(self, data):
        # carline, long or short version, as feature
        data['Carline'] = data['FIN'].astype(str).str.slice(0, 3)


    def create_time_features(self, data):
        batch,_=data.shape
        year_week=list(data.columns)
        week=list(map(lambda x: int(x[-2:]),year_week))
        week_period = 52 / (2 * np.pi)
        dow_norm=[i/week_period for i in week]
        dow = np.stack([np.cos(dow_norm), np.sin(dow_norm)], axis=-1) # => n_time*2
        dow_feature=np.tile(dow,(batch,1,1)) # => batch_size * n_time *2
        return dow_feature

    def create_auto_correlation_feature(self,data,start_ids,stop_ids):
        _, n_time=data.shape
        raw_year_autocorr = self.batch_autocorr(data, int(365.25 // 7), start_ids, stop_ids, 1.5)
        year_unknown_pct = np.sum(np.isnan(raw_year_autocorr)) / len(raw_year_autocorr)
        raw_quarter_autocorr = self.batch_autocorr(data, int(round(365.25 / 4 // 7)), start_ids, stop_ids, 2)
        quarter_unknown_pct = np.sum(np.isnan(raw_quarter_autocorr)) / len(raw_quarter_autocorr)

        logging.info("Undefined autocorr = yearly:%.3f, quarterly:%.3f" % (year_unknown_pct, quarter_unknown_pct))
        year_autocorr = self._normalize(np.nan_to_num(raw_year_autocorr))
        quarter_autocorr = self._normalize(np.nan_to_num(raw_quarter_autocorr))  # => [batch_size]

        auto_corr_features = np.stack([quarter_autocorr, year_autocorr])
        auto_corr_features=np.expand_dims(auto_corr_features.T,1)
        auto_corr_features=np.tile(auto_corr_features,(1,n_time,1))
        return auto_corr_features

    def create_median_as_feature(self,data):
        _, n_time = data.shape
        popularity = np.mean(data,axis=1,keepdims=True)
        mean=np.mean(popularity,axis=-1,keepdims=True)
        std=np.mean(popularity,axis=-1,keepdims=True)
        popularity = (popularity - mean) / (std+epsilon)
        popularity=np.expand_dims(popularity,1)
        popularity=np.tile(popularity,(1,n_time,1))
        return popularity

    def mask_short_series(self,data,start_idx,end_idx,threshold):
        issue_mask=(end_idx-start_idx)/data.shape[1]<threshold
        print("Masked %d issues from %d" % (issue_mask.sum(),len(data)))
        inv_mask=~issue_mask
        data=data[inv_mask]
        return data

    def batch_autocorr(self,data,lag,starts,ends,threshold,backoffdet=0):
        # auto correlation as feature  => n_lag*n_issue
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
        new_indexs=np.array(new_indexs).T
        return new_indexs

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

    def _normalize(self, values):
        return (values - values.mean()) / (np.std(values) + epsilon)