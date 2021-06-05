import pandas as pd
import numpy as np
import pickle
df=pd.read_csv(r'C:\Users\Sreenivasulu\Documents\changed.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder
def encoding(feature):
    df[feature+'_name_encoded'] = LabelEncoder().fit_transform(df[feature])
features = ['Processor Name','Processor Type','Operating System Type','Disk Drive','Company','Graphic Card','Touchscreen']
for i in features:
    encoding(i)
df_model = df[['Processor Name_name_encoded',
                   'Processor Type_name_encoded', 'Operating System Type_name_encoded',
                   'Disk Drive_name_encoded', 'Company_name_encoded',
                   'Graphic Card_name_encoded', 'Touchscreen_name_encoded', 'Generation', 'RAM_GB', 'DDR_Version',
                   'Price', 'Size(Inches)']]
X = df_model.drop('Price',axis=1)
Y = df_model['Price'].values
df_model
df_model.corr()
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(X_train,Y_train)
np.mean(cross_val_score(rf,X_train,Y_train,scoring='neg_mean_absolute_error',cv=3))
ypred_rf = rf.predict(X_test)
print(mean_absolute_error(Y_test,ypred_rf))
print(r2_score(ypred_rf,Y_test))
ypred_rf
with open('rf_pickle','wb') as f:
    pickle.dump(rf,f)
with open('rf_pickle','rb') as f:
    mp=pickle.load(f)
