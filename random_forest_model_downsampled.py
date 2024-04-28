import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


df = pd.read_csv('C:/Users/GGPC/Desktop/Python Sentiment Analysis/input/Results.csv')
print (df.head())


class_counts = df['Score'].value_counts()
minority_class_count = class_counts.min()

downsampled_dfs = []

# Iterate over each class in 'Score' and under-sample to match the minority class count
for score_value in class_counts.index:
    df_class = df[df['Score'] == score_value]

    if len(df_class) > minority_class_count:
        df_class_downsampled = resample(
            df_class,
            replace=False,  
            n_samples=minority_class_count,  
            random_state=7
        )
        downsampled_dfs.append(df_class_downsampled)
    else:
        downsampled_dfs.append(df_class)

# Combine the under-sampled DataFrames to create a balanced dataset
df_balanced = pd.concat(downsampled_dfs)


X = df_balanced.drop('Score',axis='columns')
y = df_balanced.Score

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators= 40)
model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)
