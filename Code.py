#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

#reading dataframe
df=pd.read_csv('creditcard.csv')
df.head()

#checking for null values
df.isnull().sum()

#counting number of fraud and no fraid values
print('Fraud:',df['Class'].value_counts()[1],'\n','No Fraud:',df['Class'].value_counts()[0])
#Creating sub-sumple with equal no. of Fraud and NO Fraud to have normal distribution of classes
df=df.sample(frac=1)
fraud=df[df['Class']==1]
Nfraud=df[df['Class']==0][:492]
newDf=pd.concat([fraud,Nfraud]).sample(frac=1,random_state=5)
newDf.head()

print(newDf['Class'].value_counts()/len(newDf))
sns.countplot(data=newDf,x='Class')
plt.show()

#scaling features
x=newDf.drop('Class',axis=1)
y=newDf['Class']
scaler=StandardScaler()
x=scaler.fit_transform(x)

#spliting data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)

#applying Logistic regression and evaluating model
LR=LogisticRegression()
LR.fit(x_train,y_train)
y_pred=LR.predict(x_test)
print(classification_report(y_test,y_pred))

#testing model on new values
values=[[6,-0.594286082,0.256157196,-0.213192213,-0.27452613,1.67859866,2.722818061,1.370145128,0.841084443,-0.492047587,-0.4104304234,-0.705145587,-0.110452262,-0.286253632,0.07435536,-0.32878305,-0.2100892268,-0.499767969,0.118764861,0.370328167,0.152735669	,-0.0734251,-0.268091632,-0.20423267,1.011591802,0.37320468,-0.384157308,0.011747356,0.14240433,101.3
]]
values=scaler.transform(values)
prediction=LR.predict(values)
if(prediction==0):
    print('No Fraud')
else:
    print('Fraud')
