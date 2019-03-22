import os
import zipfile
from six.moves import urllib
import requests
from bs4 import BeautifulSoup
from prettytable import *

DATA_URL='https://archive.ics.uci.edu/ml/machine-learning-databases/00481/EMG_data_for_gestures-master.zip'

class Create_EMG(object):
    def __init__(self):
        pass

    def download_EMG_data(self,data_dir):
        os.makedirs(data_dir, exist_ok=True)
        data_file = os.path.join(data_dir, 'EMG_data_for_gestures-master')

        if not os.path.exists(data_file):
            temp_file, _ = urllib.request.urlretrieve(DATA_URL)
            zipf = zipfile.ZipFile(temp_file)
            zipf.extractall(data_dir)
            print(zipfile.ZipFile(temp_file).namelist())
            os.remove(temp_file)

    def load_and_transfer_data(self,data_dir,out_file):
        with open(data_dir, 'r', encoding='utf-8') as temp_f:
            with open(out_file, 'w') as data_f:
                for line in temp_f:
                    print(line)
                    line = line.strip()
                    line = line.replace(', ', ',')
                    if not line or ',' not in line:
                        continue
                    if line[-1] == '.':
                        line = line[:-1]
                    line += '\n'
                    data_f.write(line)


class Create_fund(object):
    def __init__(self):
        pass

    def get_url(self,url, params=None, proxies=None):
        rsp = requests.get(url, params=params, proxies=proxies)
        rsp.raise_for_status()
        return rsp.text

    def get_fund_data(self,code, start='', end=''):
        record = {'Code': code}
        url = 'http://fund.eastmoney.com/f10/F10DataApi.aspx'
        params = {'type': 'lsjz', 'code': code, 'page': 1, 'per': 65535, 'sdate': start, 'edate': end}
        html = self.get_url(url, params)
        soup = BeautifulSoup(html, 'html.parser')
        records = []
        tab = soup.findAll('tbody')[0]
        for tr in tab.findAll('tr'):
            if tr.findAll('td') and len((tr.findAll('td'))) == 7:
                record['Date'] = str(tr.select('td:nth-of-type(1)')[0].getText().strip())
                record['NetAssetValue'] = str(tr.select('td:nth-of-type(2)')[0].getText().strip())
                record['ChangePercent'] = str(tr.select('td:nth-of-type(4)')[0].getText().strip())
                records.append(record.copy())
        return records

    def demo(self,code, start, end):
        table = PrettyTable()
        table.field_names = ['Code', 'Date', 'NAV', 'Change']
        table.align['Change'] = 'r'
        records = self.get_fund_data(code, start, end)
        for record in records:
            table.add_row([record['Code'], record['Date'], record['NetAssetValue'], record['ChangePercent']])
        return table


if __name__ == "__main__":
    print(demo('110022', '2019-01-22', '2019-03-09'))
