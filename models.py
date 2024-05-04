## this file contains the models we are experimenting with
## This should be: Randomforest, XGboost, LGBMClassifier and the NN
## each model should have the same fit, predict fucntion and be able to output uncertainties
from sklearn.multioutput import MultiOutputClassifier # for stacking Lgbmboost abd xgboost
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.src.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Lambda
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, roc_curve, auc, hamming_loss
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Normalization, Activation, BatchNormalization, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

def loadData(file="data/AlaskaClean.csv"):
    data = pd.read_csv(file)
    # Selecting features and labels
    features = ['water', 'wetland', 'shrub', 'dshrub', 'dec', 'mixed', 'spruce', 'baresnow', 'elev_m']
    labels = ['AMPI', 'AMRO', 'ATSP', 'DEJU', 'FOSP', 'GCSP', 'HETH', 'OCWA', 'ROPT', 'SAVS', 'WCSP', 'WIPT', 'WISN', 'WIWA', 'YRWA']

    # Splitting the data into training and testing sets based on the year
    train_data = data[data['year'] < 2008]
    test_data = data[data['year'] == 2008]

    # Extracting features and labels for training and testing
    X_train = np.array(train_data[features])
    y_train = np.array(train_data[labels])
    X_test = np.array(test_data[features])
    y_test = np.array(test_data[labels])

    # Show the first few rows of training features and labels to verify
    return  X_train, y_train, X_test, y_test

def getModels():
    # simple method to get all the models in a list
    rf_model = RandomForestClassifier(n_estimators=100, random_state=22)
    xgb_model = XGBClassifier(use_label_encoder=False, multi_strategy="multi_output_tree", random_state=22)
    lgb_model = MultiOutputClassifier(lgb.LGBMClassifier(verbosity=-1, random_state=22))
    #nn_model = getNN()
    models = [rf_model, xgb_model, lgb_model] #, nn_model]
    return models

class MCDropout(Layer):
    def __init__(self, rate):
        super(MCDropout, self).__init__()
        self.rate = rate

    def call(self, inputs, training=None):
        return tf.nn.dropout(inputs, rate=self.rate, name="MCDropout")

def getNN():
    normalizer = Normalization()
    model = Sequential([
        Input(shape=(9,)),
        normalizer,
        Dense(512, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Activation('relu'),
        MCDropout(0.5),
        Dense(512, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Activation('relu'),
        MCDropout(0.5),
        Dense(15, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy', AUC(name='auc')])
    return model
# Load data

