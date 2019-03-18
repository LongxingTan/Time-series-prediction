import numpy as np
import pandas as pd

class Diff(object):
    def __init__(self,series,diff_value):
        self.series=pd.Series(series)
        self.diff_value=diff_value

    def produce_diff(self):
        return self.series.diff(self.diff_value)

    def restore_diff(self):
        diff = self.diff_value
        ser2 = self.produce_diff()
        before = self.series.iloc[:diff]
        ser_ = pd.Series()
        for i in range(diff):
            after = self.series.iloc[i] + ser2[i::diff].cumsum()
            ser_ = ser_.append(after).replace(np.nan, self.series.iloc[i])
        ser_ = ser_.sort_index()
        return ser_