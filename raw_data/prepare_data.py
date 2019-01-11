import pandas as pd
import csv


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
    def import_examples(self,data_dir):
        samples=self._read_excel(data_dir[0],skiprows=1)
        failures=self._read_excel(data_dir[1],skiprows=1)
        print(samples.head(5))
        print(failures.head(5))


prepare_denza().import_examples(['./production_2018.xlsx','./failure_2018.xlsx'])