import os
import pandas as pd
import numpy as np
import datetime
from sklearn import linear_model
from scipy.special import erfinv
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
np.random.seed(7)
def load_data():
    '''Load the input Excel files from AQUA, for necessary template you could ask Longxing'''
    starttime = datetime.datetime.now()
    print("-------------Data Loading--------------")
    production_data,repair_data = pd.DataFrame(),pd.DataFrame()
    Production_path = os.path.join(os.getcwd(), '01-Production')
    Repair_path=os.path.join(os.getcwd(),'02-Repair')

    dateparse = lambda x: pd.datetime.strptime(str(x), '%m/%d/%Y')
    for file in os.listdir(Production_path):
        File_name = os.path.join(Production_path, file)
        Date_col=['Production date', 'Initial registration date','Engine production date']
        File_data = pd.read_excel(File_name, parse_dates=Date_col,date_parser=dateparse)
        production_data = production_data.append(File_data, ignore_index=True)
    for file in os.listdir(Repair_path):
        File_name=os.path.join(Repair_path,file)
        Date_col=['Production date', 'Initial registration date','Repair date','Engine production date','Date of credit (history)']
        File_data=pd.read_excel(File_name,parse_dates=Date_col,date_parser=dateparse)
        repair_data=repair_data.append(File_data,ignore_index=True)
    print("Samples|", min(production_data['Production date']), "->", max(production_data['Production date']),
          "|:", str(production_data.shape[0]))
    print("Failures|", min(repair_data['Production date']), "->", max(repair_data['Production date']),
          "|:", str(repair_data.shape[0]))
    endtime = datetime.datetime.now()
    print("Loading time [s]:", (endtime - starttime).seconds)
    return production_data, repair_data


def preprocessing_data(production_data, repair_data_,ByMileage, ByComplaints=['Fault location']):
    '''Obtain the necessary information, carline,MIS,
    return the production data and repair data, and duplicated repair'''
    print("----------Data Preprocessing-----------")
    repair_data=repair_data_.copy()
    repair_data = repair_data.loc[repair_data['Total costs'] > 0, :]
    repair_data.loc[:,'Carline'] = repair_data.loc[:,'FIN'].astype(str).str.slice(0, 3)
    repair_data.loc[:,'Repair month'] = repair_data.loc[:,'Repair date'].apply(lambda x: x.strftime('%Y%m'))
    repair_data.loc[:,'Production month'] = repair_data.loc[:,'Production date'].apply(lambda x: x.strftime('%Y%m'))
    repair_data.loc[:,'Service days'] = repair_data.loc[:,'Repair date'] - repair_data.loc[:,'Initial registration date']
    repair_data.loc[:,'Service days'] = list(map(lambda x: x.days, repair_data.loc[:,'Service days']))
    repair_data.loc[repair_data['Service days'] < 0, 'Service days'] = 0
    repair_data['Service months'] = (repair_data['Service days'] - 15) // 30 + 1
    repair_data['Mileage'] = ((repair_data['Vehicle mileage in km'] - 500) // 1000 + 1) * 1000
    duplicated_repair = repair_data.loc[repair_data.duplicated(subset=['FIN', 'Fault location'], keep=False), :]
    repair_data = repair_data.loc[~repair_data.duplicated(subset=['FIN', 'Fault location'], keep='first'), :]
    repair_data['Engine production week'] = list(map(lambda x: x.strftime("%Y%V"), repair_data['Engine production date']))
    repair_data['Production week'] = list(map(lambda x: x.strftime("%Y%V"), repair_data['Production date']))
    repair_data['Document delta'] = repair_data['Date of credit (history)'] - repair_data['Repair date']
    repair_data['Document delta'] = list(map(lambda x: x.days, repair_data['Document delta']))
    repair_data['Dealer visit start']=pd.to_datetime(repair_data['Workshop visit'].apply(lambda x: x.split(' - ')[0]),format='%d.%m.%Y')
    repair_data['Dealer visit end'] = pd.to_datetime(repair_data['Workshop visit'].apply(lambda x: x.split(' - ')[1]),format='%d.%m.%Y')
    repair_data['Dealer visit delta']=repair_data['Dealer visit end']-repair_data['Dealer visit start']
    repair_data['Dealer visit delta']=list(map(lambda x: x.days, repair_data['Dealer visit delta']))
    if ByComplaints==["Fault location"]:
        repair_data['Complaints'] = repair_data['Fault location'] + '|' + repair_data['Unnamed: 11']
    elif ByComplaints==["Damage code"]:
        repair_data['Complaints'] = repair_data['Fault location'] +repair_data['Fault type']+ '|' + repair_data['Unnamed: 11']+" "+repair_data['Unnamed: 13']
    else:
        print("Invalid complaints input type: only 'Fault location' or 'Damage code' is valid")
    print('Repair data done :)')
    production_data['Carline'] = production_data['FIN'].astype(str).str.slice(0, 3)
    production_data['Production month'] = production_data['Production date'].apply(lambda x: x.strftime('%Y%m'))
    production_data.loc[:,'Registration delta'] = production_data.loc[:,'Initial registration date'].dt.date - production_data.loc[:,'Production date'].dt.date
    production_data['Registration delta'] = list(map(lambda x: x.days, production_data['Registration delta']))
    production_data.loc[production_data['Registration delta'] < 0, 'Registration delta'] = 0
    production_data.loc[:,'Registration delta'] = Monte_cola(production_data.loc[:,'Registration delta'],bycountry=False)
    Asof_day = max(repair_data['Repair date'])
    production_data['Service days'] = (Asof_day - production_data['Production date'])
    production_data['Service days'] = list(map(lambda x: x.days, production_data['Service days']))
    production_data['Service days'] = production_data['Service days'] - production_data['Registration delta']
    production_data=production_data.merge(repair_data.loc[:,['FIN','Dealer visit delta']],how='left',on='FIN')
    production_data.fillna({'Dealer visit delta':0},inplace=True)
    production_data['Service days']=production_data['Service days']-production_data['Dealer visit delta']
    production_data.loc[production_data['Service days'] < 0, 'Service days'] = 0
    production_data['Service months'] = (production_data['Service days'] - 15) // 30 + 1
    production_data['Engine production week'] = pd.Series(production_data['Engine production date']).dt.strftime("%Y%V")
    production_data['Production week'] = pd.Series(production_data['Production date']).dt.strftime("%Y%V")
    print('Production data done :)')
    if ByMileage== ['False']:
        pass
    else:
        try:
            production_data['Mileage day'] = production_data.loc[:, 'Last repair Date (W&G)'] - production_data[
                'Production date']
            production_data['Mileage day'] = production_data.loc[
                ~pd.isnull(production_data['Mileage day']), 'Mileage day'].apply(lambda x: x.days)
            production_data.loc[:, 'Mileage day'] = production_data['Mileage day'] - production_data['Registration delta']
            production_data.loc[(production_data['Mileage day'] < 20) | (production_data['Mileage at last repair date (W&G)'] < 100), 'Mileage day'] = None
            production_data['Daily mileage'] = production_data['Mileage at last repair date (W&G)'] / production_data['Mileage day'] * 365
            mu, sigma = Lognormal(ByMileage[1], ByMileage[2])
            production_data.loc[production_data['Daily mileage'].isnull(), 'Daily mileage'] = np.random.lognormal(mu,sigma,len(production_data.loc[production_data['Daily mileage'].isnull(),:]))
            production_data.loc[:,'Daily mileage']=production_data['Daily mileage']//365
            production_data['Total mileage'] = production_data['Service days'] * production_data['Daily mileage']
        except:
            print("The valid quantile of 50% & 90% yearly mileage looks like: ['Lognormal',20715,42610]")
    #production_data.to_csv('production_list_new.csv',sep=';',index=False)
    #repair_data.to_csv('repair_list_new.csv',sep=';',index=False)
    return production_data, repair_data, duplicated_repair

def processing_data():
    '''get the data by received complained month and MIS
    return the operating month list and calendar month list'''
    pass

def ERI_component(production_data,repair_data,Condition, Predictmonth,Lowest_failure_size=10,istesting=True):
    '''weibull regression and RNN and combination'''
    print("----------Weibull Regression-----------")
    Kaplan_meier_sum, issue_list_sum, Calendar_month_predict_sum = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    repair_sum = repair_data.groupby(Condition + ['Complaints'])
    for group in tqdm(repair_sum.groups):
        repair_ele = repair_sum.get_group(group)
        if repair_ele.shape[0] > Lowest_failure_size:
            production_ele, repair_ele = Production_period_choose(production_data, repair_ele)
            if Condition:
                for con in Condition:
                    con_repair = set(repair_ele[con])
                    production_ele = production_ele[(production_ele[con].isin(con_repair))]
            production_sum = production_ele.groupby(Condition +['Service months']).size().reset_index(name='Production')
            if Condition:
                production_sum['Production cum'] = production_sum.sort_values(by='Service months', ascending=False).groupby(Condition)['Production'].cumsum()
            else:
                production_sum['Production cum'] = production_sum.sort_values(by='Service months', ascending=False)['Production'].cumsum()
            repair_sum_ele = repair_ele.groupby(Condition + ['Service months', 'Complaints']).size().reset_index(name='Failures')
            Kaplan_meier=Kaplan_meier_fun(production_sum, repair_sum_ele, Condition)
            issue_list,Calendar_month_predict,Kaplan_meier_sum=weibull_prediction(Kaplan_meier,Kaplan_meier_sum,Predictmonth,Condition)
            if issue_list.shape[0] != 0:
                issue_list_sum = pd.concat([issue_list_sum, issue_list], axis=0)
                Calendar_month_predict_sum=pd.concat([Calendar_month_predict_sum,Calendar_month_predict],axis=0)
        else:
            continue
    Kaplan_meier_sum.to_csv("MIS.csv", sep=';', index=False)
    issue_list_sum.to_csv("Issue_list.csv", sep=';', index=False)
    Calendar_month_predict_sum.to_csv("Calendar_predict.csv",sep=';',index=False)
    if issue_list_sum.empty:
        print('WARN: Too less failures to implement ERI component')
    else:
        print("-------ERI component is Done :)--------")



def ERI_system(production_data, repair_data,ByMileage,isengine=False):
    if ByMileage==['False']:
        print('Must have a mileage distribution for system level ERI')
        pass
    if isengine==True:
        ERI_sys = production_data.groupby('Engine production week', as_index=False)['Total mileage'].agg(
            {'Production': 'size', 'Mileage': np.sum})
        ERI_sys_issue = repair_data.groupby('Engine production week')['Engine production week'].size().reset_index(
            name='Failures')
        ERI_sys = ERI_sys.merge(ERI_sys_issue, how='left', on='Engine production week').fillna(0)
        ERI_sys['Engine production week_'] = ERI_sys['Engine production week'].apply(lambda x: first_monday(int(x[:4]), int(x[-2:])))
    else:
        ERI_sys = production_data.groupby('Production week', as_index=False)['Total mileage'].agg(
            {'Production': 'size', 'Mileage': np.sum})
        ERI_sys_issue = repair_data.groupby('Production week')['Production week'].size().reset_index(
            name='Failures')
        ERI_sys = ERI_sys.merge(ERI_sys_issue, how='left', on='Production week').fillna(0)
        ERI_sys['Production week_'] = ERI_sys['Production week'].apply(lambda x: first_monday(int(x[:4]), int(x[-2:])))

    failures_week, Invisible_failures=Failure_simulate(repair_data,Failure_plot=False)
    ERI_sys['Failures_portion']=ERI_sys['Failures'].apply(lambda x: x/np.sum(ERI_sys['Failures']))
    ERI_sys['Failures_sim']=round(Invisible_failures*(ERI_sys['Failures_portion']))
    ERI_sys['Failures_all']=ERI_sys['Failures']+ERI_sys['Failures_sim']
    ERI_sys['50%_pph'] = pow(ByMileage[1], 1) * sp.stats.chi2.ppf(0.5, 2 * (ERI_sys['Failures_all'] + 1)) / (
    2 * pow(ERI_sys['Mileage'], 1)) * 100
    ERI_sys['5%_pph'] = pow(ByMileage[1], 1) * sp.stats.chi2.ppf(0.05, 2 * (ERI_sys['Failures_all'] + 1)) / (
    2 * pow(ERI_sys['Mileage'], 1)) * 100
    ERI_sys['95%_pph'] = pow(ByMileage[1], 1) * sp.stats.chi2.ppf(0.95, 2 * (ERI_sys['Failures_all'] + 1)) / (
    2 * pow(ERI_sys['Mileage'], 1)) * 100
    ERI_sys.to_csv("ERI system.csv",sep=';',index=False)

    print("--------ERI system is Done :)----------")

def Failure_simulate(repair_data,Failure_plot=True):
    a_out, Kappa_out, loc_out, Lambda_out = sp.stats.exponweib.fit(repair_data['Document delta'], floc=0)
    repair_data['Repair week'] = list(map(lambda x: x.strftime("%Y%U"), repair_data['Repair date']))
    failures_week = repair_data.groupby('Repair week')['Repair week'].size().reset_index(name='Value')
    failures_week['Failure coefficient'] = [1 - sp.stats.exponweib.cdf(7 * x + 3, 1, a_out, loc=loc_out, scale=Lambda_out) for x in range(len(failures_week))[::-1]]
    Invisible_failures = np.ceil(failures_week['Value'].dot(failures_week['Failure coefficient']))
    if Failure_plot:
        x = np.linspace(0, 100, 1000)
        pdf_fitted = sp.stats.exponweib.pdf(x, a=a_out, c=Kappa_out, loc=loc_out,scale=Lambda_out)
        plt.plot(x, pdf_fitted, 'r-')
        plt.hist(repair_data['Document delta'], bins=30, normed=True, alpha=.3)
        plt.savefig('Delta simulation between repair and document.jpg')
    return failures_week,Invisible_failures

def Lognormal(x1,x2,y1=0.5,y2=0.9):
    mu=np.log(x1)
    sigma=np.sqrt(2)/2*(np.log(x1)-np.log(x2))/(erfinv(2*y1-1)-erfinv(2*y2-1))
    return mu,sigma

def Weibull_coordinate(x, y):
    Weibull_x = np.log(x)
    Weibull_y = np.log(-np.log(1 - y))
    return Weibull_x, Weibull_y

def LR(x, y):
    clf = linear_model.LinearRegression()
    if len(x)>36:
        clf.fit(x[:36], y[:36])  # only the first three years data attend the prediction due to Mercedes warranty policy
    else:
        clf.fit(x[:-2], y[:-2])
    score = clf.score(x, y)
    return clf.coef_[0], clf.intercept_, score, clf

def weibull_prediction(Kaplan_meier,Kaplan_meier_sum,Predictmonth, Condition):
    '''predict the testing data according to the train model'''
    issue_list, Calendar_month_predict = pd.DataFrame(), pd.DataFrame()
    Kaplan_meier_sum = pd.concat([Kaplan_meier_sum, Kaplan_meier], axis=0)
    maxmonth = Kaplan_meier_sum['Service months'].max() + Predictmonth + 1
    maxmonthele = Kaplan_meier['Service months'].max() + 1
    x, y = Kaplan_meier['Service months'].values, Kaplan_meier['Failure probability'].values
    if Kaplan_meier.dropna(axis=0, how='any').shape[0]>= 3:
        Weibull_x, Weibull_y = Weibull_coordinate(x.reshape(-1, 1), y.reshape(-1, 1))
        slope, intercept, score, clf = LR(Weibull_x.reshape(-1, 1), Weibull_y.reshape(-1, 1))
        issue_list = Kaplan_meier.pivot_table(index=Condition + ['Complaints'], columns=['Service months'],
                                              values='Failure probability').reset_index()
        issue_list['Slope'] = slope
        issue_list['Intercept'] = intercept
        issue_list['Score'] = score
        Available_vehicles = Create_availablevehicles(Kaplan_meier['Service months'],
                                                      Kaplan_meier['Production cum'],
                                                      maxmonth, Predictmonth)
        Failure_probability = Create_failureprobability(Kaplan_meier['Service months'],
                                                        Kaplan_meier['Failure probability'],
                                                        maxmonth)
        for i in range(int(maxmonthele), int(maxmonth)):
            value = 1 - np.exp(-np.exp(clf.predict(np.log(i))))
            issue_list[i] = value
            Failure_probability.iloc[i - 1] = value
        Available_vehicles_delta = pd.DataFrame(0, index=np.arange(len(Failure_probability)-1), columns=np.arange(Predictmonth-1))
        Failure_probability_delta = []
        for i in range(0, Predictmonth-1):
            Available_vehicles_delta.iloc[:, i] = Available_vehicles.iloc[:, i + 1] - Available_vehicles.iloc[:, i]
        for i in range(0,len(Failure_probability)-1):
            Failure_probability_delta.append(Failure_probability[i + 1] - Failure_probability[i])
        Calendar_month_predict = pd.DataFrame(np.dot(Available_vehicles_delta.T, Failure_probability_delta)).T
        for i in ['Complaints', 'Slope'] + Condition:
            Calendar_month_predict.ix[:, i] = issue_list.ix[:, i]
    return issue_list, Calendar_month_predict,Kaplan_meier_sum

def train_test_split(repair_data, test_size=0.5):
    '''split the train and test data by some date
    return the train_'''
    repair_month = sorted(list(set(repair_data['Repair month'])))
    repair_month_train = repair_month[:np.ceil(len(repair_month) * test_size)]
    repair_month_test = repair_month[np.ceil(len(repair_month) * test_size):]
    repair_data_train=repair_data.loc[repair_data['Repair moth'].isin(repair_month_train),:]
    repair_data_test = repair_data.loc[repair_data['Repair moth'].isin(repair_month_test), :]
    return repair_data_train, repair_data_test

def weibull_evaluation():
    '''evaluate the training error and testing error'''
    pass

def weibull_visualization():
    '''visualize the model and prediction result for report'''
    pass

def Production_period_choose(production_data_, repair_data_):
    '''Initial check about affected production month'''
    production_production=set(production_data_['Production month'])
    repair_production=set(repair_data_['Production month'])
    Intersection_production=production_production&repair_production
    production_data=production_data_.loc[production_data_['Production month'].isin(Intersection_production),:]
    repair_data=repair_data_.loc[repair_data_['Production month'].isin(Intersection_production),:]
    return production_data, repair_data

def Monte_cola(Registration,bycountry=False):
    '''use monte cola method to simulate the missing registration date'''
    Regis=Registration.copy()
    Monte_cola_boolen=Regis.notnull()
    Monte_cola_list=list(Regis[Monte_cola_boolen])
    Regis[~Monte_cola_boolen]=[np.random.choice(Monte_cola_list) for _ in range(len(Regis[~Monte_cola_boolen]))]
    return Regis

def Kaplan_meier_fun(production_sum, repair_sum_ele,Condition):
    Kaplan_meier = pd.merge(production_sum, repair_sum_ele, how='left',
                            on=(Condition + ['Service months'])).fillna(0)
    try:
        Kaplan_meier['Complaints'] = list(set(Kaplan_meier['Complaints']) - {0})[0]
    except:
        pass
    Kaplan_meier = Kaplan_meier.loc[Kaplan_meier['Service months'] > 0, :]
    Kaplan_meier['Survival probability interval'] = 1 - Kaplan_meier['Failures'] / Kaplan_meier['Production cum']
    Kaplan_meier['Failure probability'] = 1 - Kaplan_meier['Survival probability interval'].cumprod()
    Kaplan_meier = Kaplan_meier.loc[Kaplan_meier['Failure probability'] > 0, :]
    Kaplan_meier = Kaplan_meier[np.isfinite(Kaplan_meier['Failure probability'])]
    return Kaplan_meier


def Create_availablevehicles(Service_months,Production_cum,maxmonth, Predictmonth):
    '''Create the available sample of each service months for the future calender month'''
    asof_sell=max(Production_cum)
    Avg_sell=asof_sell//max(Service_months)
    Available_vehicles=pd.DataFrame()
    Available_vehicles['Service months']=range(1,int(maxmonth))
    Current_vehicle=pd.concat([Service_months,Production_cum],axis=1)
    Available_vehicles=Available_vehicles.merge(Current_vehicle,how='left',on='Service months').fillna(0)
    for i in range(Predictmonth+1):
        Available_vehicles[i]=np.roll(Available_vehicles.loc[:,'Production cum'],i)
        for j in range(i):
            Available_vehicles.ix[j,i]=(i-j)*Avg_sell+asof_sell
    return Available_vehicles.iloc[:,2:]

def Create_failureprobability(Service_months,Failure_probability_,maxmonth):
    '''Create the failure probability'''
    Failure_probability=pd.DataFrame()
    Failure_probability['Service months']=range(1,int(maxmonth))
    Failure_probability_ = pd.concat([Service_months, Failure_probability_], axis=1)
    Failure_probability=Failure_probability.merge(Failure_probability_,on='Service months',how='left').fillna(0)
    return Failure_probability.iloc[:,-1]

def first_monday(year, week):
    try:
        d = datetime.date(year, 1, 4)  # The Jan 4th must be in week 1  according to ISO
        return d + datetime.timedelta(weeks=(week-1), days=-d.weekday())
    except:
        return 'NA'
