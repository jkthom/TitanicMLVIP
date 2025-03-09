import sklearn.model_selection
import sklearn.feature_extraction
import pandas as pd
import random
import numpy as np

random.seed(0)

train_data = pd.read_csv('train.csv')
survived = train_data['Survived']
train_data = train_data.drop('Survived', axis=1)
train_data.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
train_data.set_index(keys=['PassengerId'], drop=True, inplace=True)

train_nan_map = {'Age': train_data['Age'].mean(), 'Fare': train_data['Fare'].mean(), 'Embarked': train_data['Embarked'].mode()[0]}

train_data.fillna(value=train_nan_map, inplace=True)

train_data = pd.get_dummies(train_data, columns=['Embarked'], dtype=int)

columns_map = {'Sex': {'male': 0, 'female': 1}}
train_data.replace(columns_map, inplace=True)
train_data =  np.hstack( (train_data, survived.values.reshape((-1,1 ) ) ) )

#train_data = train_data[[c for c in train_data if c not in ['Survived']] + ['Survived']]
kf = sklearn.model_selection.KFold(n_splits=5)

for i, (train_index, test_index) in enumerate(kf.split(train_data)):
    #train_data.iloc[train_index, :].to_csv('train_' + str(i) + '.csv.gz', compression='gzip', index=False)
    #train_data.iloc[test_index, :].to_csv('test_' + str(i) + '.csv.gz', compression='gzip', index=False)
    
    np.savetxt('train_' + str(i) + '.csv.gz', train_data[train_index], delimiter=',')
    np.savetxt('test_' + str(i) + '.csv.gz', train_data[test_index], delimiter=',')

