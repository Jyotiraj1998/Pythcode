# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:49:40 2020

@author: shamaun
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\shamaun\Desktop\Datasets\maintenance_data.csv")

#data Analysis

sns.countplot(data.team)

sns.countplot(data.team[data.broken==1])
plt.show()

#swarmplot 

sns.swarmplot(data.team,data.lifetime,
              hue=data.broken)
plt.show()

sns.swarmplot(data.provider,data.lifetime,
              hue=data.broken)
plt.show()

sns.distplot(data.lifetime[data.broken==1])
sns.distplot(data.lifetime[data.broken==0])
plt.legend(['broken','not broken'])
plt.show()

sns.distplot(data.pressureInd[data.broken==1])
sns.distplot(data.pressureInd[data.broken==0])
plt.legend(['broken','not broken'])
plt.show()

sns.distplot(data.temperatureInd[data.broken==1])
sns.distplot(data.temperatureInd[data.broken==0])
plt.legend(['broken','not broken'])
plt.show()

sns.distplot(data.moistureInd[data.broken==1])
sns.distplot(data.moistureInd[data.broken==0])
plt.legend(['broken','not broken'])
plt.show()

#selecting inputs

ip = data.drop(['broken','pressureInd',
                'temperatureInd','moistureInd'],
               axis=1)

op = data.broken

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=(['team',OneHotEncoder(),[1]],
                                     ['provider',OneHotEncoder(),[2]]),
                       remainder='passthrough')


ip = ct.fit_transform(ip)



from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts = train_test_split(ip,op, test_size=0.2)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(xtr)
xtr = sc.transform(xtr)
xts = sc.transform(xts)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

LR = LogisticRegression()
KN = KNeighborsClassifier(n_neighbors=5)
SV = SVC()

LR.fit(xtr,ytr)

KN.fit(xtr,ytr)

SV.fit(xtr,ytr)

LR.score(xts,yts)
KN.score(xts,yts)
SV.score(xts,yts)

from sklearn.metrics import confusion_matrix

predLR = LR.predict(xts)
predKN = KN.predict(xts)
predSV = SV.predict(xts)


print('''confusion matrix of Logistic regression'''+'\n',
      confusion_matrix(yts, predLR),'\n'
      +'***'.center(10,'~'))


print('''confusion matrix of KNN'''+'\n',
      confusion_matrix(yts, predKN),'\n'
      +'***'.center(10,'~'))


print('''confusion matrix of SVC'''+'\n',
      confusion_matrix(yts, predSV),'\n'
      +'***'.center(10,'~'))

"""
new_value = mean_of_column - existing_value 
            ------------------------------
                  STD_of_column
"""

