import pandas as pd
import numpy as np
import math
import time
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import itertools
import warnings
warnings.filterwarnings("ignore")


class main():

	def __init__(self):
		print("reading train.csv,test.csv and store.csv and putting them in pandas dataframe")
		self.test=pd.read_csv("test.csv")
		self.train=pd.read_csv("train.csv")
		self.store=pd.read_csv("store.csv")
		self.pre_processing_store()
		self.pre_processing_train()
		self.test_train_date_and_merge(self.train)
		self.test_train_date_and_merge(self.test)
		self.X_train, self.X_test = train_test_split(self.train, test_size=0.2)
		self.train=shuffle(self.train)
		self.features = list(self.train.columns)
		self.features.remove('Sales')
		self.X=self.X_train[self.features]
		self.Y=self.X_train['Sales']
		#self.parameter_tuning_xgboost(self,train)
		self.train_xgboost_model_save_and_pickel()


	def pre_processing_store(self):
		print("Pre-processing store file")
		print("Fill all Null values with -1")
		self.store.fillna(-1,inplace=True)
		print("label encoding StoreType")
		self.store.loc[self.store['StoreType']=='a', 'StoreType']='1'
		self.store.loc[self.store['StoreType']=='b', 'StoreType']='2'
		self.store.loc[self.store['StoreType']=='c', 'StoreType']='3'
		self.store.loc[self.store['StoreType']=='d', 'StoreType']='4'
		self.store['StoreType'] = self.store['StoreType'].astype(float)
		print("label encoding Assortment")
		self.store.loc[self.store['Assortment']=='a', 'Assortment']='1'
		self.store.loc[self.store['Assortment']=='b', 'Assortment']='2'
		self.store.loc[self.store['Assortment']=='c', 'Assortment']='3'
		self.store['Assortment'] = self.store['Assortment'].astype(float)
		print("Droping promotinterval")
		self.store.drop('PromoInterval',axis=1,inplace=True)


	def pre_processing_train(self):
		print("Pre-processing train")
		print("Considering only the stores which are opne and sales greater than 0")
		self.train = self.train[(self.train['Open']==1)&(self.train['Sales']>0)]
		print("Label encoding stateHoliday")
		self.train.loc[self.train['StateHoliday']=='a', 'StateHoliday']=1
		self.train.loc[self.train['StateHoliday']=='b', 'StateHoliday']=1
		self.train.loc[self.train['StateHoliday']=='c', 'StateHoliday']=1
		self.train.loc[self.train['StateHoliday']=='0', 'StateHoliday']=0
		#train['StateHoliday']=train['StateHoliday'].astype(float)
		print("We need not do the same for test as it contains only 1 value 0")


	def test_train_date_and_merge(self,process):
		print("Seperating day, month and year from date")
		for ds in [process]:
		    tmpDate = [time.strptime(x, '%Y-%m-%d') for x in ds.Date]
		    ds[  'mday'] = [x.tm_mday for x in tmpDate]
		    ds[  'mon'] = [x.tm_mon for x in tmpDate]
		    ds[  'year'] = [x.tm_year for x in tmpDate]
		process.drop('Date',axis=1,inplace=True)
		print("performing left outer join withs store")
		process = process.merge(self.store, on = 'Store', how = 'left')

	def rmspe(self,y, yhat):
	    return np.sqrt(np.mean(((y - yhat)/y) ** 2))

	def rmspe_xg(self,yhat, y):
	    y = np.expm1(y.get_label())
	    yhat = np.expm1(yhat)
	    return "root_mean_square_error", rmspe(y, yhat)

	def parameter_tuning_xgboost(self):
		print("parameter tuning with xgboost")
		model = XGBRegressor()
		param_grid = {
		'max_depth':range(8,16,2),
		'min_child_weight':range(7,15,2),
		'learning_rate':[x/10 for x in range(1,4)],
		'subsample':[x/10 for x in range(5,9)],
		'colsample_bytree':[x/10 for x in range(5,9)]
		}
		grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=3)
		grid_result = grid_search.fit(self.X,self.Y)
		print("best parameters are:")
		print(grid_search.best_params_)
		test_pred=grid_result.predict(self.X_test[self.features])
		error = rmspe(self.test_pred, self.X_test['Sales'])
		print('error on using parameter tuned xgboost', error)
		

	def train_xgboost_model_save_and_pickel(self):
		print("predicting result with best parameters of xgboost")
		xgb_reg = XGBRegressor(max_depth = 14, learning_rate=0.1,min_child_weight = 11, subsample = 0.8, colsample_bytree = 0.7)
		xgb_reg=xgb_reg.fit(self.train[self.features], self.train['Sales'])
		test_pred=xgb_reg.predict(self.test[self.features])
		d = {'Id': test.Id, 'Sales': test_pred}
		output = pd.DataFrame(data=d)
		print("saving as xgboost_parameter_tuned.csv")
		output.to_csv('xgboost_parameter_tuned.csv',index=False)
		print("making pickel file")
		xgboost_pkl = open("xgboost_pickel_file.pkl", "wb")
		pickle.dump(test_pred, xgboost_pkl)
		xgboost_pkl.close()
		
		


	def Linear_regression(self):
		lin_reg=LinearRegression()
		lin_reg=lin_reg.fit(self.X, self.Y)
		test_pred=lin_reg.predict(self.X_test[self.features])
		test_set=self.X_test['Sales']
		error = self.rmspe(test_pred, test_set)
		print('error on using linear Regression', error)

	def Decision_tree_regressor(self):
		from sklearn.tree import DecisionTreeRegressor
		dtree_reg=DecisionTreeRegressor()
		dtree_reg=dtree_reg.fit(self.X, self.Y)
		test_pred=dtree_reg.predict(self.X_test[self.features])
		test_set=self.X_test['Sales']
		error = self.rmspe(test_pred, test_set)
		print('error on using Decision Tree Regressor', error)

	def Random_forest_regressor(self):
		from sklearn.ensemble import RandomForestRegressor
		ranfor_reg=RandomForestRegressor()
		ranfor_reg=ranfor_reg.fit(self.X, self.Y)
		test_pred=ranfor_reg.predict(self.X_test[self.features])
		test_set=self.X_test['Sales']
		error = self.rmspe(test_pred, test_set)
		print('error on using Random Forest Regressor', error)

	def models_with_default_parameters(self):
		print("predicting different models with their default parameters")
		self.Linear_regression()
		self.Decision_tree_regressor()
		self.Random_forest_regressor()


class Visualization():
	
	def __init__(self):
		self.str_to_date = lambda x: pd.datetime.strptime(x, "%Y-%m-%d")
		self.train = pd.read_csv("train.csv",sep=',', parse_dates=['Date'], date_parser=self.str_to_date,low_memory = False)
		self.store = pd.read_csv("store.csv", low_memory = False)

	def plotting(self):
		print "The Train dataset has {} Rows and {} Variables".format(str(self.train.shape[0]),str(self.train.shape[1]))
		print ("The Store dataset has {} Rows (which means unique Shops) and {} Variables\n".format(str(self.store.shape[0]),str(self.store.shape[1]))) 
		print ("-Over those two years, {} is the number of times that different stores closed on given days.".format(self.train[(self.train.Open == 0)].count()[0]))
		print ("\n")
		print ("-From those closed events, {} times occured because there was a school holiday. " .format(self.train[(self.train.Open == 0) & (self.train.SchoolHoliday == 1)&(self.train.StateHoliday == '0') ].count()[0]))
		print ("\n")
		print ("-And {} times it occured because of either a bank holiday or easter or christmas.".format(self.train[(self.train.Open == 0) &
				 ((self.train.StateHoliday == 'a') |
				  (self.train.StateHoliday == 'b') | 
				  (self.train.StateHoliday == 'c'))].count()[0]))
		print ("\n")
		print ("-But interestingly enough, {} times those shops closed on days for no apparent reason when no holiday was announced.".format(self.train[(self.train.Open == 0) &
				 (self.train.StateHoliday == "0")&(self.train.SchoolHoliday == 0)].count()[0]))
		self.train=self.train.drop(self.train[(self.train.Open == 0) & (self.train.Sales == 0)].index)
		self.train = self.train.reset_index(drop=True) 
		print ("Our new training set has now {} rows ".format(self.train.shape[0]))
		self.train=self.train.drop(self.train[(self.train.Open == 1) & (self.train.Sales == 0)].index)
		self.train = self.train.reset_index(drop=True) 
		fig, axes = plt.subplots(1, 2, figsize=(17,3.5))
		axes[0].boxplot(self.train.Sales, showmeans=True,vert=False)
		axes[0].set_xlim(0,max(self.train["Sales"]+1000))
		axes[0].set_title('Boxplot For Sales Values')
		axes[1].hist(self.train.Sales, cumulative=False, bins=20)
		axes[1].set_title("Sales histogram")
		axes[1].set_xlim((min(self.train.Sales), max(self.train.Sales)))
		fig, axes = plt.subplots(1, 2, figsize=(17,3.5))
		axes[0].boxplot(self.train.Customers, showmeans=True,vert=False)
		axes[0].set_xlim(0,max(self.train["Customers"]+100))
		axes[0].set_title('Boxplot For Customer Values')
		axes[1].hist(self.train.Customers, cumulative=False, bins=20)
		axes[1].set_title("Customers histogram")
		axes[1].set_xlim((min(self.train.Customers), max(self.train.Customers)))
		self.store_check_distribution=self.store.drop(self.store[pd.isnull(self.store.CompetitionDistance)].index)
		fig, axes = plt.subplots(1, 2, figsize=(17,3.5))
		axes[0].boxplot(self.store_check_distribution.CompetitionDistance, showmeans=True,vert=False,)
		axes[0].set_xlim(0,max(self.store_check_distribution.CompetitionDistance+1000))
		axes[0].set_title('Boxplot For Closest Competition')
		axes[1].hist(self.store_check_distribution.CompetitionDistance, cumulative=False, bins=30)
		axes[1].set_title("Closest Competition histogram")
		axes[1].set_xlim((min(self.store_check_distribution.CompetitionDistance), max(self.store_check_distribution.CompetitionDistance)))
		{"Mean":np.nanmean(self.store.CompetitionDistance),"Median":np.nanmedian(self.store.CompetitionDistance),"Standard Dev":np.nanstd(self.store.CompetitionDistance)}
		self.train_store = pd.merge(self.train, self.store, how = 'left', on = 'Store')
		self.train_store['SalesperCustomer']=self.train_store['Sales']/self.train_store['Customers']
		fig, axes = plt.subplots(2, 3,figsize=(17,10) )
		palette = itertools.cycle(sns.color_palette(n_colors=4))
		plt.subplots_adjust(hspace = 0.28)
		axes[0,0].bar(self.store.groupby(by="StoreType").count().Store.index,self.store.groupby(by="StoreType").count().Store,color=[next(palette),next(palette),next(palette),next(palette)])
		axes[0,0].set_title("Number of Stores per Store Type \n ")
		axes[0,1].bar(self.train_store.groupby(by="StoreType").sum().Sales.index,self.train_store.groupby(by="StoreType").sum().Sales/1e9,color=[next(palette),next(palette),next(palette),next(palette)])
		axes[0,1].set_title("Total Sales per Store Type (in Billions) \n ")
		axes[0,2].bar(self.train_store.groupby(by="StoreType").sum().Customers.index,self.train_store.groupby(by="StoreType").sum().Customers/1e6,color=[next(palette),next(palette),next(palette),next(palette)])
		axes[0,2].set_title("Total Number of Customers per Store Type (in Millions) \n ")
		axes[1,0].bar(self.train_store.groupby(by="StoreType").sum().Customers.index,self.train_store.groupby(by="StoreType").Sales.mean(),color=[next(palette),next(palette),next(palette),next(palette)])
		axes[1,0].set_title("Average Sales per Store Type \n Fig 1.4")
		axes[1,1].bar(self.train_store.groupby(by="StoreType").sum().Customers.index,self.train_store.groupby(by="StoreType").Customers.mean(),color=[next(palette),next(palette),next(palette),next(palette)])
		axes[1,1].set_title("Average Number of Customers per Store Type \n ")
		axes[1,2].bar(self.train_store.groupby(by="StoreType").sum().Sales.index,self.train_store.groupby(by="StoreType").SalesperCustomer.mean(),color=[next(palette),next(palette),next(palette),next(palette)])
		axes[1,2].set_title("Average Spending per Customer in each Store Type \n ")
		self.train_store['Month']=self.train_store.Date.dt.month
		self.train_store['Year']=self.train_store.Date.dt.year
		sns.factorplot(data = self.train_store, x ="Month", y = "Sales", col = 'Promo', hue = 'Promo2', row = "Year" ,sharex=False)
		plt.show()

if __name__ == "__main__":
	v=Visualization()
	v.plotting()
	m=main()
	m.models_with_default_parameters()
	