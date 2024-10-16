# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/028cfac5-3f28-4389-9a00-a18ef00350b1)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/77b17051-1b4c-482c-8331-4c0376db5f81)

```
max_val = np.max(np.abs(df[['Height', 'Weight']]))
max_val
```
![image](https://github.com/user-attachments/assets/0b235b14-d87b-4bd1-a775-0b4f5b026982)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/bcce8490-a228-4b33-aa8e-3f0682d0c0c1)

```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/36670d66-e97e-445f-9de7-94ebb66fa14f)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/e4ef4f9f-d254-4478-8415-d63977c78bde)

```

df1=pd.read_csv("bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df1[['Height','Weight']]=scaler.fit_transform(df1[['Height','Weight']])
df1
```
![image](https://github.com/user-attachments/assets/f59b2fc9-97d1-416a-999a-83dd11ce7151)

```

df2=pd.read_csv("bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2.head()
```
![image](https://github.com/user-attachments/assets/8786d218-a4bd-4d0d-bd03-2aee0691471f)

```
import pandas as pd
import numpy as np
import seaborn as sns
​
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
​
data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/57969894-a4ac-4079-9a01-c7704c59d522)

```

data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/0810899c-afb6-41a9-a889-8caef919f1f2)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/b02c7889-fae3-47ce-a981-0cac1d8a5ee4)

```

data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/a44312da-5d62-461b-acaf-64fe55f3b885)

```

sal=data["SalStat"]
​
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
```
![image](https://github.com/user-attachments/assets/4341bd28-f2f7-415d-8cdb-9a99a26595fc)

```


sal2=data2['SalStat']
​
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/530870ee-55bd-4795-a5b9-35b4cb00957d)
```

data2
```
![image](https://github.com/user-attachments/assets/a1be4f54-8e37-44e7-bbec-fc789b721099)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/89c49648-a62e-440c-a227-775572c95823)
```

columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/7a3e4b36-66b2-410a-9080-f4c6b61b5a14)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/b35c4576-5a55-4c1f-8b7f-368496b9c6c9)

```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/28dfb950-8499-42f0-ac43-da34d09b2274)

```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/e93cfedc-b1e4-437a-b9a1-40aa5970206b)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)

prediction=KNN_classifier.predict(test_x)
​
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/6ef90cee-a508-475b-b928-1856ac95366b)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/98051fc0-0537-4219-8a48-0fd15f8e0a10)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/e22e1bb8-9935-441d-9af4-bf74a577e51b)

```
data.shape
```
![image](https://github.com/user-attachments/assets/cab332cb-2183-4937-b483-7870437c00d4)


```

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}
​
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
​
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
​
selected_feature_indices=selector.get_support(indices=True)
​
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/5c9e8f7e-44df-42cf-affe-f9c0f152b25f)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
​
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/88368550-3778-42f6-b1b5-9cb35146406e)
```
tips.time.unique()
```
![image](https://github.com/user-attachments/assets/41e780b0-806a-4c9f-b66b-0302b062a16a)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

```

![image](https://github.com/user-attachments/assets/00b76f53-e8fe-45d7-b721-ec0effc2c977)
```

chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")

```
![image](https://github.com/user-attachments/assets/dad7654b-0c54-4d3a-8d4a-27424b43da71)

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
