import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('./data.csv')

# get dataset details
print(dataset.head(5))
print(dataset.columns.values)
print(dataset.info())
print(dataset.describe())

# data cleansing
X = dataset.iloc[:, 2:32]
print(X.info())
print(type(X))
y = dataset.iloc[:, 1]
print(y)

'''encode the labels to 0, 1 respectively'''
print(y[100:110])
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print([y[100:110]])

# lets split dataset now
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
scalar = StandardScaler()
XTrain = scalar.fit_transform(XTrain)
XTest = scalar.transform(XTest)

# choosing hyper parameters
'''
def classifier(optimizer):
    model = Sequential()
    model.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=30))
    model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=classifier)
params = {'batch_size': [1, 5], 'epochs': [100, 120], 'optimizer': ['adam', 'rmsprop']}
gridSearch = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=10)
gridSearch = gridSearch.fit(XTrain, yTrain)
score = gridSearch.best_score_
bestParams = gridSearch.best_params_
print(score)
print(bestParams)
'''

# modeling
model = Sequential()
model.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=30))
model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(XTrain, yTrain, batch_size=1, epochs=120)
model.save('./cancer_model.h5')
yPred = model.predict(XTest)
yPred = [1 if y > 0.5 else 0 for y in yPred]
matrix = confusion_matrix(yTest, yPred)
print(matrix)
accuracy = (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
print("Accuracy: " + str(accuracy * 100) + "%")
