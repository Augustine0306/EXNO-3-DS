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
 import pandas as pd
 df=pd.read_csv("/content/Encoding Data.csv")
 df
      ![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/d6c38ae3-7ac6-4be0-a73c-63a488cd1be9)

      from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/855edb19-e377-4721-90f0-29dc5acc73e8)

df['bo2']=e1.fit_transform(df[["ord_2"]])
df
![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/c93526c0-ec0f-4855-b7ab-7f850a5636a1)

df['bo2']=e1.fit_transform(df[["ord_2"]])
df
![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/4e34382d-c2eb-4335-bdcf-575a9984b7cf)

le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/aa3cb67f-ffab-4f3f-888c-dc80bfb54772)

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/580ad1d4-2d7b-49d1-a8d6-05d378945e45)

pd.get_dummies(df2,columns=["nom_0"])

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/4759ad8f-7814-4d06-9827-78ab1140844c)

pip install --upgrade category_encoders

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/93b7df8c-5d9c-4748-b3ae-e0c5d8da4df2)

from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
fb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/d808d470-5a64-4fd3-b703-390c5f62c706)

from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/5ed393af-f756-4ef5-964b-282e921ad23b)

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/1310e5f1-5b10-4771-81f8-6df78b0ecbb8)

df.skew()

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/8f326e15-ef37-49cf-b5ba-25b7853cd4c2)

np.log(df["Highly Positive Skew"])

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/937cfbc5-b13d-4465-bbd0-da1aadf639ad)

np.reciprocal(df["Moderate Positive Skew"])

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/fe183eb4-1f5e-4f34-a1e2-2a08f651a813)


np.sqrt(df["Highly Positive Skew"])

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/53d6337c-285d-4198-bee3-e5e9a657c5b3)

np.square(df["Highly Positive Skew"])

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/4f454b7b-25a2-441d-b105-c6b0c3e287a2)

df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"]) 
df

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/6ed7c0ae-38ea-42a9-a5c1-0dfa1e50ccfa)

df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/18f1ddc0-7d98-43fc-95ca-3efb65908c4a)

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/0448784d-eeae-491f-9cf6-5a34e478fe59)

import matplotlib.pyplot as plt import seaborn as sns import statsmodels.api as sm import scipy.stats as stats

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/3c445cdb-f2f6-425a-9d20-98ef8aedf2a7)

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/a9474364-442d-460c-b13f-a1430fb43821)

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

![image](https://github.com/Nishanth-018/EXNO-3-DS/assets/149347651/7c7cfafb-e1d6-4583-961c-1ba4dcdc4235)

# RESULT:
           Hence performing Feature Encoding and Transformation process is Successful.

       


       
