import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.models import Sequential
from keras.activations import relu, sigmoid
from keras.layers import Input

df_bank = pd.read_csv('Churn_Modelling.csv')

X = df_bank.iloc[:, 3:13]
y = df_bank.iloc[:, 13]

geography = pd.get_dummies(X['Geography'], drop_first=True)
gender = pd.get_dummies(X['Gender'], drop_first=True)

X = pd.concat([X, geography, gender], axis=1)
X = X.drop(['Geography', 'Gender'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
sc_xtrain = sc.fit_transform(x_train)
sc_xtest = sc.transform(x_test)

# Adding 3 layers to the network and increasing accuraccy and using 'he_normal'
# Adding 1st layers (10 hidden neurons)
classifier = Sequential()
classifier.add(Dense(units = 10, kernel_initializer = 'he_normal', activation = 'relu', input_dim = 11))
classifier.add(Dropout(0.3))
# Adding 2nd layers (20 hidden neurons)
classifier.add(Dense(units=20, kernel_initializer = 'he_normal', activation = 'relu'))
classifier.add(Dropout(0.4))
# Adding 3rd layers (15 hidden layers)
classifier.add(Dense(units=15, kernel_initializer = 'he_normal', activation = 'relu'))
classifier.add(Dropout(0.2))
# Adding 1 output layer
classifier.add(Dense(units=1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

# compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = (['accuracy']))

model_history = classifier.fit(sc_xtrain, y_train, validation_split=0.33, batch_size=10, epochs = 10)

joblib.dump(classifier, 'model_app/classifier.pkl')

print("Model trained and saved successfully.")