## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```python
import pandas as pd
df=pd.read_csv('/content/Encoding Data.csv')
df
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/5ce33461-bc04-4453-b779-2a370989e584)
## ORDINAL ENCODER
```python
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/673da170-5ec7-469d-87ca-65915419cce9)
```python
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/f1099ad0-0793-45b1-a29a-0130434b96c4)
## LABEL ENCODER
```python
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/cea3639f-adf5-4b72-ac79-75092b020f2f)
## ONEHOT ENCODER
```python
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
```
```python
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/0ebfaaaf-db6c-436a-8e14-d5a503ad9237)
```python
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/35efd046-c8cf-4e16-ba4c-baeeb7d92d4b)
## BINARY ENCODER
```python
pip install --upgrade category_encoders
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/e21c5fb7-dadd-4b35-a261-11f7bd8a9af4)
```python
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/771f6710-241b-4a69-98ca-8cf443905d8f)
```python
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/3f64b8e5-20ea-42e4-97b0-906621a40ad1)
## TARGET ENCODER
```python
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/95528dcc-9871-4c93-9c3b-99f591b8714d)
## DATA TRANSFORMATION
```python
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv('/content/Data_to_Transform.csv')
df
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/6a8d74d2-d19d-41f6-8450-821771056910)
```python
df.skew()
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/9b7a6f36-4ae4-4b05-a7af-6b3ee6955244)
```python
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/4b68b805-c67e-4f14-a433-ef98156496d8)
```python
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/ad335c36-6370-4f1e-86a9-e73941c9b66c)
```python
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/3ac142a7-13de-43c9-af8c-7d92e8e4cb2c)
```python
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/82c5e20c-8106-4257-a918-a9b85f8a5771)
```python
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/c36f7785-85c6-4414-adb8-338e4ba2dd6a)
```python
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/9eee06e1-6f2d-4001-be60-5b1adf5d6318)
```python
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/1ba3ebcf-279e-4f40-b02b-db5ff48bc125)
```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/324e2604-e438-4e4e-9224-aaadf9c6dacb)
```python
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/6ae37f06-725b-4e2c-89bd-be6f7d3c6781)
```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/6bcb1913-9714-4e60-abf7-776ced014e31)
```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/2bc747ad-9344-4911-b42c-55aff96c37c4)
```python
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/813159fc-6480-4a92-adf3-fca72740336f)
```python
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/7fbcb366-653d-45e0-96ce-3256137565b3)
```python
dt=pd.read_csv("/content/titanic_dataset.csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/a27ed54c-a8cb-45d9-a559-12883484beec)
```python
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```
![image](https://github.com/DHINESH-SEC/EXNO-3-DS/assets/119404460/6dad177e-cc85-4dfd-83d9-bc2b5d787df3)



# RESULT:
Thus perform Feature Encoding and Transformation process is executed successfully.

       
