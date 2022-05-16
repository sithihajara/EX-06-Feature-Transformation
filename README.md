# EX-06-Feature-Transformation

## AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file. 

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

 
# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set
### STEP 4
Save the data to the file

# CODE
## Data_To_Transform.csv
```
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats 

df=pd.read_csv("Data_To_Transform.csv")  

df

df.skew()  

#FUNCTION TRANSFORMATION:  
#Log Transformation  
np.log(df["Highly Positive Skew"]) 

#Reciprocal Transformation  
np.reciprocal(df["Moderate Positive Skew"]) 

#Square Root Transformation  
np.sqrt(df["Highly Positive Skew"]) 

#Square Transformation  
np.square(df["Highly Negative Skew"]) 

#POWER TRANSFORMATION:  
#Boxcox method:
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"]) 
df

#Yeojohnson method:
df["Moderate Positive Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Positive Skew"]) 
df

df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"]) 
df

df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"]) 
df

#QUANTILE TRANSFORMATION:  

pip install scikit-learn

import sklearn

from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal') 


df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])  
sm.qqplot(df['Moderate Negative Skew'],line='45')  
plt.show()

sm.qqplot(df['Moderate Negative Skew_1'],line='45')  
plt.show()  

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])  
sm.qqplot(df['Highly Negative Skew'],line='45')  
plt.show() 

sm.qqplot(df['Highly Negative Skew_1'],line='45')  
plt.show()  

df["Moderate Positive Skew_1"]=qt.fit_transform(df[["Moderate Positive Skew"]])  
sm.qqplot(df['Moderate Positive Skew'],line='45')  
plt.show()  

sm.qqplot(df['Moderate Positive Skew_1'],line='45')  
plt.show() 

df["Highly Positive Skew_1"]=qt.fit_transform(df[["Highly Positive Skew"]])  
sm.qqplot(df['Highly Positive Skew'],line='45')  
plt.show() 

sm.qqplot(df['Highly Positive Skew_1'],line='45')  
plt.show()

df.skew() 
df
```

# OUPUT
## import package and reading the data set
<img width="465" alt="impo" src="https://user-images.githubusercontent.com/93427278/168627654-7e57920b-ee4e-4ca1-8573-3343876fecc0.png">

## Function Transformation

<img width="488" alt="fun1" src="https://user-images.githubusercontent.com/93427278/168628081-f9e40819-6480-49ad-b2f0-8d63b90ee061.png">
<img width="418" alt="fun2" src="https://user-images.githubusercontent.com/93427278/168628103-86511968-b869-423b-8327-cd90b7175832.png">

## POWER TRANSFORMATION

<img width="499" alt="pow1" src="https://user-images.githubusercontent.com/93427278/168628193-31455475-18cf-4594-a273-db7f61dffb29.png">
<img width="497" alt="pow2" src="https://user-images.githubusercontent.com/93427278/168628221-c1993a76-251e-4a18-98f0-a3d884ef940a.png">
<img width="500" alt="pow3" src="https://user-images.githubusercontent.com/93427278/168628234-0d998062-42e8-4a48-9fd7-ee272474c145.png">
<img width="494" alt="pow4" src="https://user-images.githubusercontent.com/93427278/168628245-6fae00ec-3f3d-4c66-ba24-1927311fd149.png">

## Quantile Transformation

<img width="459" alt="qua1" src="https://user-images.githubusercontent.com/93427278/168628779-0b367b93-dfe9-4aa1-b79d-f995e14e701a.png">
<img width="485" alt="qua2" src="https://user-images.githubusercontent.com/93427278/168628796-6d1b47c3-ba64-46fe-a499-6f1b6acb4a62.png">
<img width="488" alt="qua3" src="https://user-images.githubusercontent.com/93427278/168628816-3c4ba440-7894-4b8b-97b7-1ae40d35e111.png">
<img width="485" alt="qua4" src="https://user-images.githubusercontent.com/93427278/168628874-22aa79d8-e97b-4242-ace5-b719985a5fc6.png">
<img width="466" alt="qua5" src="https://user-images.githubusercontent.com/93427278/168628891-e71dd74c-b07b-48a7-9cc5-727e0a1f0d89.png">

## Final result

<img width="622" alt="finalr" src="https://user-images.githubusercontent.com/93427278/168629029-571375e0-4eac-4f4d-a85a-1bcae99c55cc.png">

# CODE
## titanic_dataset.csv
```
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats  

df=pd.read_csv("titanic_dataset.csv")  
df  

df.drop("Name",axis=1,inplace=True)  
df.drop("Cabin",axis=1,inplace=True)  
df.drop("Ticket",axis=1,inplace=True) 

df.isnull().sum()  

df["Age"]=df["Age"].fillna(df["Age"].median())  
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])  
df

from sklearn.preprocessing import OrdinalEncoder  

embark=["C","S","Q"]  
emb=OrdinalEncoder(categories=[embark])  
df["Embarked"]=emb.fit_transform(df[["Embarked"]])
df

#FUNCTION TRANSFORMATION:  
#Log Transformation  
np.log(df["Fare"])  

#ReciprocalTransformation  
np.reciprocal(df["Age"]) 

#Squareroot Transformation:  
np.sqrt(df["Embarked"]) 

#POWER TRANSFORMATION:  
#Boxcox method:
df["Age _boxcox"], parameters=stats.boxcox(df["Age"])  
df 

df["Pclass _boxcox"], parameters=stats.boxcox(df["Pclass"])    
df 

#Yeojohnson method:
df["Fare _yeojohnson"], parameters=stats.yeojohnson(df["Fare"])  
df 

df["SibSp _yeojohnson"], parameters=stats.yeojohnson(df["SibSp"])  
df 

df["Parch _yeojohnson"], parameters=stats.yeojohnson(df["Parch"])  
df  

#QUANTILE TRANSFORMATION  
from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891) 

df["Age_1"]=qt.fit_transform(df[["Age"]])  
sm.qqplot(df['Age'],line='45')  
plt.show() 

sm.qqplot(df['Age_1'],line='45')  
plt.show()  

df["Fare_1"]=qt.fit_transform(df[["Fare"]])  
sm.qqplot(df["Fare"],line='45')  
plt.show() 

sm.qqplot(df['Fare_1'],line='45')  
plt.show()

df.skew()  
df
```

# OUTPUT
## titanic_dataset.csv
### import package and reading the data set

<img width="494" alt="impot" src="https://user-images.githubusercontent.com/93427278/168630902-f3351954-a76b-43de-90d9-c2d0ce505ef1.png">

## Data Cleaning Process:

<img width="428" alt="darace1" src="https://user-images.githubusercontent.com/93427278/168630972-ff4bb30f-eadd-499a-97ca-00bed5b52930.png">
<img width="443" alt="darace2" src="https://user-images.githubusercontent.com/93427278/168630996-e779dede-1934-4d8c-9202-ef6cfc76f03f.png">

## FUNCTION TRANSFORMATION:

<img width="512" alt="fun1" src="https://user-images.githubusercontent.com/93427278/168631098-e404ec03-c275-4de0-bba6-6724b568693d.png">
<img width="407" alt="fun2" src="https://user-images.githubusercontent.com/93427278/168631115-07d8b0a5-1c61-4ee4-87e4-050ce10fd8c0.png">

## POWER TRANSFORMATION:

<img width="449" alt="pow1" src="https://user-images.githubusercontent.com/93427278/168631453-71278c9e-c891-45e1-b557-224d9d515f96.png">
<img width="479" alt="pow2" src="https://user-images.githubusercontent.com/93427278/168631469-8e61b80a-d2e4-4302-9b37-42ac74d64026.png">
<img width="497" alt="pow3" src="https://user-images.githubusercontent.com/93427278/168631485-95358951-b2eb-43c7-b947-c21a1cbbfe53.png">
<img width="611" alt="pow4" src="https://user-images.githubusercontent.com/93427278/168631511-718c3a51-74a3-42cc-9351-26a1623b8dab.png">
<img width="601" alt="pow5" src="https://user-images.githubusercontent.com/93427278/168631518-0e9fd164-3ff9-4b1c-82cb-c86799fa43b6.png">

## QUANTILE TRANSFORMATION

<img width="405" alt="qua1" src="https://user-images.githubusercontent.com/93427278/168631700-7eedfaf3-bc96-480c-9a6e-c181f243e809.png">
<img width="380" alt="qua2" src="https://user-images.githubusercontent.com/93427278/168631733-04047907-c8a4-4a21-b10d-5a218fe8cc8a.png">
<img width="325" alt="qua3" src="https://user-images.githubusercontent.com/93427278/168631806-e13fb8fb-a7d6-4050-bb4e-4e7f8d961954.png">

# Final result

<img width="615" alt="fnrace1" src="https://user-images.githubusercontent.com/93427278/168631853-4fb52139-89eb-4a8b-b107-a64e85caa4b7.png">

<img width="613" alt="fnrace2" src="https://user-images.githubusercontent.com/93427278/168631861-a7458a7b-b748-4323-8ff8-fe2df0f52167.png">

# RESULT:
Hence, Feature transformation techniques is been performed on given dataset and saved into a file successfully.
