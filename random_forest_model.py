import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv('C:/Users/GGPC/Desktop/Python Sentiment Analysis/input/Results.csv')
print (df.head())


X = df.drop('Score',axis='columns')
y = df.Score

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators= 10)
model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)




#Hyperparameter testing if wanted is provided below

# class_counts = df['Score'].value_counts()
# print(class_counts)

#n_estimators= 50, min_samples_split= 4, min_samples_leaf= 4, max_depth= 10,

# from sklearn.model_selection import RandomizedSearchCV

# param_grid = {
#     'n_estimators': [20, 30, 50, 100, 200],  # More trees
#     'max_depth': [10, 20, 30],  # Limiting tree depth
#     'min_samples_split': [2, 4, 8],  # Splits with more samples
#     'min_samples_leaf': [1, 2, 4],  # Leaf nodes with more samples
#     'class_weight' : [class_weights]
# }

# random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=10, cv=5)
# random_search.fit(X_train, y_train)

# best_model = random_search.best_estimator_

# print("Best hyperparameters (random search):", random_search.best_params_)
# print("Best accuracy:", best_model.score(X_test, y_test))


