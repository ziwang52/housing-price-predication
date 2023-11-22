#!/usr/bin/env python
# coding: utf-8

# In[1]:


#project for housing price prediction
#zi wang
#CSCI6397

#jupyterlink http://localhost:8888/notebooks/ml%20project/Untitled.ipynb


import pandas as pd 
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[2]:


#Data set get from https://www.kaggle.com/datasets/vedavyasv/usa-housing
data = pd.read_csv("USA_Housing.csv" )
data.head(5)


# In[3]:


#understand the detail of dataset
data.info()
display(data.shape)


# In[4]:


data.describe()


# In[5]:


#show the correlation between variables
data.corr()


# In[6]:


# draw attributes' graph
#cited from https://www.kaggle.com/code/suprabhatsk/usa-housing-price-prediction-for-beginners/notebook


'''
looks different and may lead to an error if run in vscode 
original code in Jupyter :
%matplotlib inline
from matplotlib import pyplot as plt
plt.style.use('ggplot')
data.hist(bins=50,figsize=(16,9))
'''

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.style.use('ggplot')
data.hist(bins=50,figsize=(16,9))


# In[7]:


#from the result of the graph, we can say the price is more correlated to avergae of the attributes expect average numbe of bedroom

#show correlation matrics in tabular format

data.corr().Price.sort_values(ascending=False)


# In[8]:


#show attributes correlation in seaborn format
import seaborn as sns
sns.pairplot(data)


# In[9]:


# from the last line of the graph, there is a linear relationship  between all attributes except average area number of bedroom
sns.displot(data.Price)


# In[10]:



#Data preprocessing
#data clean, address is not useful for the proccess
data = data.drop(['Address'], axis=1)
data.head()


# In[11]:


#split training and test data set
 
attributes=data.drop(['Price'], axis=1)
price_labels=data.Price
print(attributes.head(5))
print(price_labels.head(5))


# In[12]:




attributes.head(5)


# In[ ]:





# In[13]:


#data type transform
#code citied from https://towardsdatascience.com/predicting-house-prices-with-machine-learning-62d5bcd0d68f

# Wrapper for one hot encoder to allow labelling of encoded variables

from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X)
        d_out = pd.DataFrame(
            sparse_matrix.toarray(), columns=new_columns, index=X.index
        )
        return d_out

    def fit_transform(self, X, **kwargs):
        self.fit(X)
        return self.transform(X)

    def get_new_columns(self, X):
        new_columns = []
        for i, column in enumerate(X.columns):
            j = 0
            while j < len(self.categories_[i]):
                new_columns.append(f"{column}_<{self.categories_[i][j]}>")
                j += 1
        return new_columns

# Define funtion to encode categorrical variables with and rejoin to initial data
def transform(data, df):

    # isolate categorical features
    cat_columns = df.select_dtypes(["object"]).columns
    cat_df = df[cat_columns]

    # isolate the numeric features
    numeric_df = df.select_dtypes(include=np.number)

    # initialise one hot encoder object spcify handle unknown and auto options to keep test and train same size
    ohe = OneHotEncoder(categories="auto", handle_unknown="ignore")
    # Fit the endcoder to training data
    ohe.fit(data[cat_columns])

    # transform input data
    df_processed = ohe.transform(cat_df)

    # concatinate numeric features from orginal tables with encoded features
    df_processed_full = pd.concat([df_processed, numeric_df], axis=1)

    return df_processed_full
train_set= transform(attributes,attributes)


# In[14]:


#split 80% training and 20% testing 
from sklearn import preprocessing

x_train, x_test, y_train, y_test = train_test_split(attributes,price_labels,test_size=0.2)
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

y_standard=((y_train-y_train.mean())/(y_train.std()))
y_standard_test=((y_test-y_test.mean())/(y_test.std()))
x_test = scaler.transform(x_test)

print ("size of training data :", x_train.shape," ", y_train.shape)
print ("size of testing data :" , x_test.shape," ", y_test.shape)


# In[ ]:





# In[15]:


#calculate coefficients
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
y_train_log = np.log(y_train)

# R2(coefficient of determination) regression score

# linear_regression on test dataset
import time
from sklearn.metrics import r2_score, mean_squared_error

print("linear_regression model")

start_time = time.process_time()
linear_regression = linear_model.LinearRegression().fit(x_train,y_standard)

linear_test = r2_score(linear_regression.predict(x_test), y_standard_test)

#calculate MSE for test datasets
 
y_pred = linear_regression.predict(x_test)

mse = mean_squared_error(y_pred, y_standard_test)
 # End of fit time
print(time.process_time() - start_time, "Seconds")

print('Test Mean Squared Error:  ', mse)
print('linear_regression score: ',linear_test)
#print(x_train)
 

print('intercept for traning dataset: \n', linear_regression.intercept_)
print('coefficients for traning dataset: \n', linear_regression.coef_)

pd.DataFrame(linear_regression.coef_, index=data.columns[:-1], columns=['Values'])


# In[16]:


#ridge regression 

print("ridge regression model")

start_time = time.process_time()

ridge_model = linear_model.Ridge()
ridge_model.fit(x_train, y_standard)



ridge_r2_score = r2_score(ridge_model.predict(x_test), y_standard_test)

 # End of fit time
print(time.process_time() - start_time, "Seconds")
print('ridge regression score: ',ridge_r2_score)
print('intercept for traning dataset: \n', ridge_model.intercept_)
pd.DataFrame(ridge_model.coef_, index=data.columns[:-1], columns=[' ridge_model Values'])


# In[17]:


#decision tree model
from sklearn.tree import DecisionTreeRegressor
 
print("decision_tree_model ")

start_time = time.process_time()

tree_reg=DecisionTreeRegressor()
tree_reg.fit(x_train, y_standard)

tree_r2_score= r2_score(tree_reg.predict(x_test), y_standard_test)
   


 # End of fit time
print(time.process_time() - start_time, "Seconds")
print(' decision_tree_model score: ',tree_r2_score)


# In[18]:


# random forest regressor

from sklearn.ensemble import RandomForestRegressor


print("random forest model ")

start_time = time.process_time()

RF_reg=RandomForestRegressor()
RF_reg.fit(x_train, y_standard)

random_r2_score=r2_score(RF_reg.predict(x_test),y_standard_test)
   


 # End of fit time
print(time.process_time() - start_time, "Seconds")
print(' random forest model score: ',random_r2_score)


# In[19]:


#ransac regressor


print("RANSAC model ")

start_time = time.process_time()

ransac= linear_model.RANSACRegressor()
ransac.fit(x_train, y_standard)

ransac_r2_score = r2_score(ransac.predict(x_test),y_standard_test)
  


# End of fit time
print(time.process_time() - start_time, "Seconds")


print('RANSAC_model score: ',ransac_r2_score)


# 

# In[20]:


# GradientBoosting model
from sklearn.ensemble import GradientBoostingRegressor



print(" GradientBoostingRegressor  ")

start_time = time.process_time()

gradient_boosting=GradientBoostingRegressor()

gradient_boosting.fit(x_train, y_standard)


gradient_boosting_score=r2_score(gradient_boosting.predict(x_test),y_standard_test)
gradient_boosting_score  


 # End of fit time
print(time.process_time() - start_time, "Seconds")
print('gradient_boosting_score : ',gradient_boosting_score)


# In[21]:



result= {
    'Models':['linear regression','ridge regression ','decision tree regression ', 'random forest regressor','RANSAC regressor','GradientBoostingRegressor'],
'Scores':[linear_test,ridge_r2_score,tree_r2_score,random_r2_score,ransac_r2_score,gradient_boosting_score]
}
pd.DataFrame(result,columns=['Models','Scores']).sort_values(ascending= False, by=['Scores'])

