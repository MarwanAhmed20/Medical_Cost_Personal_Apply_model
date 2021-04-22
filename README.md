# Medical-Cost-Personal

The dataset can be downloaded from:
https://www.kaggle.com/mirichoi0218/insurance

1-load dataset 
    	First we load the dataset from Kaggle
2- data prepossessing
	In this phase preparing the data so in first:
•	Chick out for missing value in data and there aren’t 
•	transform categorical data :
encoded the sex , smoker and region rows to le_sex , le_smoker and le_region
•	One hot encoding in array that has region_encoded to 3 (region_0, region_1, region_2, region_3)
•	Finally dropped ('sex','smoker','region','region_encoded','region_0')
Now data is preparing to apply the models
•	feature scaling of the data to comes all values into same scale
3- train and test split 
 	data x and y 
	X=['age','sex_encoded','bmi','children','smoker_encoded','region_1','regi	on_2','region_3']
	Y= ['charges']
	And split to X_train, X_test, y_train, y_test.
4- apply models 
•	Decision Tree Regressor Model
 used parameter DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=10,random_state=33)
•	Linear Regression Model  
 used parameter LinearRegression(fit_intercept=True, normalize=True,copy_X=True,n_jobs=-1) 
•	Extra Trees Regressor Model
used parameter ExtraTreesRegressor(n_estimators = 200)
•	KNeighbors Regressor Model
We used parameter KNeighborsRegressor(n_neighbors = 10 , weights ='distance' ,p=2, metric='minkowski')

