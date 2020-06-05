#import PyQt5
import numpy as np
import pandas as pd
import sklearn
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier 
from pandas import Series, DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras.models  import Sequential, K
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
print(tf.__version__)

# load dataset
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("diabetes.csv", header=None, names=col_names)

feature_cols = ['glucose','bp','bmi','pedigree','age']
X = pima[feature_cols].ix[:,(0,1,2,3,4)].values
y = pima.ix[:,8].values
#Arrange
X= np.delete(X, (0), axis=0)
y = np.delete(y, (0), axis=0)
X=X.astype(float)
y=y.astype(float)

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .15, random_state=10)

#Train the model
LogReg = LogisticRegression(penalty="l2", C=1.800,class_weight='balanced')
LogReg.fit(X_train, y_train)

#Predict 
y_pred = LogReg.predict(X_test)
#Metrics
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix

print(classification_report(y_test, y_pred))

print("Validation Accuracy Logistic Regression:",metrics.accuracy_score(y_test, y_pred)*100)
#print("Precision Logistic Regression:",metrics.precision_score(y_test, y_pred)*100)



# kMeans Clustering

kmeansModel = KMeans(n_clusters = 2, algorithm = 'auto', max_iter = 900)
kmeansModel.fit(X_train)

correct = 0
for i in range(len(X_train)):
    predict_me = np.array(X_train[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeansModel.predict(predict_me)
    if prediction[0] == y_train[i]:
        correct += 1

KMeansAcc = (correct/len(X_train))*100
if KMeansAcc < 50.0:
    print('Accuracy KMeans Clustering: ', 100.0 - KMeansAcc)
else:   print('Accuracy KMeans Clustering: ',KMeansAcc)
      
# Descision Trees

DTreeModel = DecisionTreeClassifier(criterion = 'entropy',max_depth = 3)
DTreeModel.fit(X_train,y_train)
yPredDTree = DTreeModel.predict(X_test)
cDTree = 0
for indx in range(len(yPredDTree)):
    if yPredDTree[indx] == y_test[indx]:
        cDTree +=1
print('Validation Accuracy Descision Tree :{}'.format((cDTree/len(X_test))*100))


# Random Forest Classifier


rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
#print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Validation Accuracy Random Forest: {}".format((rf.score(X_test, y_test))*100))


# Custom Neural Network in Keras

#Optimizer
SGDOpt = SGD(lr = 0.01)

#Normalizing the Data

normalizer = StandardScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)

DNNModel = keras.models.Sequential([
        Dense(16,activation = 'relu',input_shape= (5,)),
        Dense(8,activation = 'relu'),
        Dense(4,activation = 'relu'),
        Dense(1,activation = 'sigmoid')
        ])
    
DNNModel.compile(optimizer = SGDOpt,loss = 'binary_crossentropy',metrics = ['acc'])

print('Training Neural Network Model for 1000 Epochs......')

NNHist = DNNModel.fit(X_train_norm,y_train,validation_data = (X_test_norm,y_test),epochs = 1000,verbose = 0)


print('Validation Accuracy Neural Network:',(NNHist.history['val_acc'][-1])*100)

fig, ax = plt.subplots()
ax.plot(NNHist.history["loss"],'r', marker='.', label="Train Loss")
ax.plot(NNHist.history["val_loss"],'b', marker='.', label="Validation Loss")
ax.legend()

from PyQt5 import QtWidgets,uic
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap


def Result():
	glu= float(dlg.Iglu.text())
	bpp= float(dlg.Ibp.text())
	bmii=float(dlg.Ibmi.text())
	pedi=float(dlg.Iped.text())
	agee=float(dlg.Iage.text())
	if(pedi==0):
		pedi=0.4726
	[pred1]=LogReg.predict([[glu,bpp,bmii,pedi,agee]])
	
	if (pred1==0):
		dlg.output.setText("No! Our Diagnostics suggest you \ndo not suffer from Diabetes \n ٩(◕‿◕)۶")
	else:
		dlg.output.setText("Yes. Our Diagnostics suggest you \ndo suffer from Diabetes. \nPlease visit a Doctor. \n (｡•́︿•̀｡)	 ")
	
def Clfn():
    dlg.Iglu.clear()
    dlg.Iped.clear()
    dlg.Ibp.clear()
    dlg.Ibmi.clear()
    dlg.Iage.clear()
    dlg.output.clear()

app= QtWidgets.QApplication([])
dlg=uic.loadUi("gui1.ui")
pixmap=QPixmap('img1.png')

dlg.img.setPixmap(pixmap)
dlg.subbtn.clicked.connect(Result)
dlg.clrbtn.clicked.connect(Clfn)

dlg.show()
app.exec()
