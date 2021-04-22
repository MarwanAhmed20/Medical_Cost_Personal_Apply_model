import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error
from sklearn import neighbors
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import statsmodels.formula.api as sm

import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


#load data
data = pd.read_csv("D:\marwan\FCI\ML\Project\insurance.csv")
data.head()



#Check out the Missing Values
#x_data.dropna(inplace=True)
#print(x_data.isnull().sum())   
#print(y_data.isnull().sum())

#replace NaN value with mean
#print(x_data.tail())
#print(x_data.replace(np.NaN,x_data['children'].mean()).tail()) 



#transform categorical data
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

data['sex_encoded'] = le_sex.fit_transform(data.sex)
data['smoker_encoded'] = le_smoker.fit_transform(data.smoker)
data['region_encoded'] = le_region.fit_transform(data.region)

#print(data.head(5) , "\n")

ohe_region = OneHotEncoder()
# One hot encoding (OHE) to array--
arr_ohe_region = ohe_region.fit_transform(data.region_encoded.values.reshape(-1,1)).toarray()

# Convert array OHE to dataframe and append to existing dataframe--
dfOneHot = pd.DataFrame(arr_ohe_region, columns=['region_'+str(i) for i in range(arr_ohe_region.shape[1])])
data = pd.concat([data, dfOneHot], axis=1)


preprocessed_data = data.drop(['sex','smoker','region','region_encoded','region_0'], axis=1)

#print(preprocessed_data.head())


#feature scaling
standard_x = StandardScaler()
x_train = standard_x.fit_transform(preprocessed_data)
x_test = standard_x.fit_transform(preprocessed_data)
#After Feature Scaling all values comes into same scale
#print(preprocessed_data.info())



variables = ['age','sex_encoded','bmi','children','smoker_encoded','region_1','region_2','region_3']

X = preprocessed_data[variables]
sc = StandardScaler()
X = sc.fit_transform(X) 
Y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=44)



#apply Decision Tree Regressor Model


DecisionTreeRegressorModel = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=10,random_state=33)
DecisionTreeRegressorModel.fit(X_train, y_train)

#Calculating Details
print('DecisionTreeRegressor Train Score is : ' , DecisionTreeRegressorModel.score(X_train, y_train))
print('DecisionTreeRegressor Test Score is : ' , DecisionTreeRegressorModel.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = DecisionTreeRegressorModel.predict(X_test)
print('Predicted Value for DecisionTreeRegressorModel is : ' , y_pred[:5])
print('Predicted Value for DecisionTreeRegressorModel is : ' , y_test[:5])

#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Absolute Error Value is : ', MAEValue)

#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Squared Error Value is : ', MSEValue)

#Calculating Median Squared Error
MdSEValue = median_absolute_error(y_test, y_pred)
print('Median Squared Error Value is : ', MdSEValue )



print("-------------------------------------")



#Applying Linear Regression Model 


LinearRegressionModel=LinearRegression(fit_intercept=True, normalize=True,copy_X=True,n_jobs=-1)
LinearRegressionModel.fit(X_train , y_train)

#Calculating Details
print('Linear Regression Train Score is : ' , LinearRegressionModel.score(X_train, y_train))
print('Linear Regression Test Score is : ' , LinearRegressionModel.score(X_test, y_test))
print('----------------------------------------------------')


#Calculating Prediction
y_pred = LinearRegressionModel.predict(X_test)

print('Predicted Value for Linear Regression is : ' , y_pred[:5])
print('Predicted Value for Linear Regression is : ' , y_test[:5])
#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
print('Mean Absolute Error Value is : ', MAEValue)

MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') 
print('Mean Squared Error Value is : ', MSEValue)

MdSEValue = median_absolute_error(y_test, y_pred)
print('Median Squared Error Value is : ', MdSEValue )


print("-------------------------------------")




#apply ExtraTreesRegressor

regressor = ExtraTreesRegressor(n_estimators = 200)
regressor.fit(X_train,y_train)

#Calculating Details
print('ExtraTreesRegressorModel Train Score is : ' , regressor.score(X_train, y_train))
print('ExtraTreesRegressorModel Test Score is : ' , regressor.score(X_test, y_test))
print('----------------------------------------------------')

#prediction and evaluation

y_pred = regressor.predict(X_test)

print('Predicted Value for ExtraTreesRegressor is : ' , y_pred[:5])
print('Predicted Value for ExtraTreesRegressor is : ' , y_test[:5])


#Calculating Mean Absolute Error

MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
print('Mean Absolute Error Value is : ', MAEValue)

MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') 
print('Mean Squared Error Value is : ', MSEValue)

MdSEValue = median_absolute_error(y_test, y_pred)
print('Median Squared Error Value is : ', MdSEValue )


print("-------------------------------------")



#apply Knn model

knn = neighbors.KNeighborsRegressor(n_neighbors = 10 , weights ='distance' ,p=2, metric='minkowski')
y_pred = knn.fit(X_train, y_train)
y_pred = knn.predict(X_test) 

#Calculating Details
print('KNeighborsRegressorModel Train Score is : ' , knn.score(X_train, y_train))
print('KNeighborsRegressorModel Test Score is : ' , knn.score(X_test, y_test))
print('----------------------------------------------------')



print('Predicted Value for Knn is : ' , y_pred[:5])
print('Predicted Value for Knn is : ' , y_test[:5])

#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
print('Mean Absolute Error Value is : ', MAEValue)

MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') 
print('Mean Squared Error Value is : ', MSEValue)

MdSEValue = median_absolute_error(y_test, y_pred)
print('Median Squared Error Value is : ', MdSEValue )




print('=====================================')
print('=====================================')
'''
#comparing
models = [DecisionTreeRegressorModel , LinearRegressionModel , regressor , knn]
x=0
for m in models:
    x+=1
    
    for n in range(2,5):
        print('result of model number : ' , x ,' for cv value ',n,' is ' , cross_val_score(m, X, Y, cv=n))  
        print('-----------------------------------')

    print('=====================================')
    print('=====================================')
    
'''

preprocessed_data['bmi_int'] = preprocessed_data['bmi'].apply(lambda x: int(x))

# data distribution analysys
print('Data distribution analysys')
for v in variables:
    preprocessed_data = preprocessed_data.sort_values(by=[v])
    preprocessed_data[v].value_counts().plot(kind = 'bar')
    plt.title(v)
    plt.show()







