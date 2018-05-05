#ERI is a method based on Weibull regression from survival analysis to predict all failures future development
from data_helper import *

def main():
    Condition = ['Carline'] #You can type "Sales country","Carline", so it looks like ['Sales country','Carline']
    Predictmonth = 12
    ByComplaints = ["Fault location"]  # you can choose ["Fault location"] or ["Damage code"]
    ByMileage=['Lognormal',22145,42010] #you can choose ["Lognormal,20176,52345"] or ['False']
    Lowest_failure_size=3

    production_data, repair_data = load_data()
    production_data, repair_data, duplicated_repair = preprocessing_data(production_data, repair_data,ByComplaints=ByComplaints, ByMileage=ByMileage)
    ERI_component(production_data, repair_data,istesting=True, Condition=Condition, Predictmonth=Predictmonth, Lowest_failure_size=Lowest_failure_size)
    ERI_system(production_data, repair_data,isengine=False,ByMileage=ByMileage)


if __name__ == "__main__":
    main()