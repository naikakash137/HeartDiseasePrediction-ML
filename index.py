import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data collection and preprocessing
heart_data=pd.read_csv('C:/Users/naika/Downloads/heart.csv')

print("Top 5 data \n",heart_data.head(5))

print("Bottom 5 data \n",heart_data.tail())

#number of rows and columns of the dataset
print("Number of rows and columns \n",heart_data.shape)

#getting info about data
print("Info about the dataset \n",heart_data.info())

# checking for missing values     (There are no missing values in the current dataset)
print("Checking the missing values \n",heart_data.isnull().sum())


# statistical measures for all the columns
print("Describe function \n",heart_data.describe())

# checking the distribution of target variable    (indicates 526 have heart disease and 499 doesnot have )
print("Checking the distribution of target variables \n",heart_data['target'].value_counts())


# splitting the features and the target
X=heart_data.drop(columns='target',axis=1)
Y=heart_data['target']

print("Qualities \n",X)

print("Target \n",Y)

# splitting the data into training and testing data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)    #random_state is used to split the data


print("X shape ",X.shape,"X train shape ", X_train.shape,"X test shape ", X_test.shape)


# Model training
model=LogisticRegression()


# training the model with LogisticRegression
model.fit(X_train, Y_train)

# Model evaluation
# Accuracy score
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


print("Accuracy on Training data : ",training_data_accuracy)


# Accuracy on test data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Accuracy on Testing data : ",testing_data_accuracy)



# Building a predictive system
input_data=(58,0,0,100,248,0,0,122,0,1,1,0,2)
# change the input data to a numpy array
input_data_as_numpy_array = np.array(input_data)
# reshape thenumpy array as we are predicting for instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
# print(prediction)

if(prediction[0] == 0):
    print("The person doesnot has heart disease")
else:
    print("The person has a heart disease")

