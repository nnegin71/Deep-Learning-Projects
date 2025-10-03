# Import libraries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa
import librosa.display

import IPython.display as ipd
from IPython.display import Audio

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


#local functions
import explore_data
import util
#---------------------------
############## Exploring dataset ###############
# Load one of the audio files, play audio, plot time-series data and different spectrograms
audio_file = "./data/fold1/101415-3-0-2.wav"
# explore_data.explore_audio(audio_file)

df = pd.read_csv('./data/UrbanSound8K.csv')
df.head()
##############
#       slice_file_name    fsID  start        end  salience  fold  classID             class
# 0    100032-3-0-0.wav  100032    0.0   0.317551         1     5        3          dog_bark
# 1  100263-2-0-117.wav  100263   58.5  62.500000         1     5        2  children_playing
# 2  100263-2-0-121.wav  100263   60.5  64.500000         1     5        2  children_playing
# 3  100263-2-0-126.wav  100263   63.0  67.000000         1     5        2  children_playing
# 4  100263-2-0-137.wav  100263   68.5  72.500000         1     5        2  children_playing
##############
explore_data.explore_data_distribution(df)

print(f"Shape of datafram is {df.shape}")
# check for missed data
print("Missed data")
df.isnull().sum()

#---------------------------
############## Data Preprocessing Pipeline ###############
# Feature Extraction
frame_size=1024
hop_length=512
ind_features = []
audio_dataset_path = "./data/"

for index_num, row in tqdm(df.iterrows(), total=len(df)):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold' + str(row["fold"]),str(row["slice_file_name"]))
    label = row['class']
    
    mfcc_feat = util.mfcc_extractor(file_name)
    zcr_feat = util.zcr_extractor(file_name,frame_size,hop_length)
    
    ind_features.append([mfcc_feat, zcr_feat, label])

feature_df= pd.DataFrame(ind_features, columns=['MFCCs', 'ZCR', 'Labels'])

feature_df.head()
##############
#                                                MFCCs       ZCR            Labels
# 0  [-211.93698, 62.581207, -122.813156, -60.74528...  0.127790          dog_bark
# 1  [-417.0052, 99.336624, -42.995586, 51.073326, ...  0.139174  children_playing
# 2  [-452.39316, 112.36253, -37.57807, 43.19586, 8...  0.102104  children_playing
# 3  [-406.47922, 91.1966, -25.043558, 42.78452, 11...  0.131091  children_playing
# 4  [-439.63867, 103.86224, -42.658787, 50.690277,...  0.118249  children_playing
##############
#drop ZCR column
new_df = feature_df.copy()
new_df.drop(columns=['ZCR'],axis = 1, inplace=True)

# Features
X=np.array(new_df['MFCCs'].tolist()) 
# Labels
y=np.array(new_df['Labels'].tolist())
# Setting up one-hot-encoding to map label classes to numerics
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Create a dict for mapping of numeric labels to original categories 
# Label mapping (Category → Encoded Value)
label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
print("Label mapping (Category → Encoded Value):")
print(label_mapping)
# Reverse mapping (Encoded Value → Category)
reverse_mapping = dict(zip(range(len(le.classes_)), le.classes_))
print("\nReverse mapping (Encoded Value → Category):")
print(reverse_mapping)
# one-hot labels
y_categorical = to_categorical(y_encoded)
y=y_categorical
print("Shape of one-hot labels:", y_categorical.shape)
print("First 5 one-hot labels:\n", y_categorical[:5])


############## Modeling ###############
# split data to train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# creating the model
num_labels = y.shape[1]
input_dim = X_train.shape[1]

model = Sequential()

model.add(Dense(units=256, input_shape=(input_dim,)))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))

# Setting optimizer, loss function, and accuracy metric
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
##############
# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ dense (Dense)                        │ (None, 256)                 │           2,816 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ batch_normalization                  │ (None, 256)                 │           1,024 │
# │ (BatchNormalization)                 │                             │                 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ activation (Activation)              │ (None, 256)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout (Dropout)                    │ (None, 256)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 512)                 │         131,584 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ batch_normalization_1                │ (None, 512)                 │           2,048 │
# │ (BatchNormalization)                 │                             │                 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ activation_1 (Activation)            │ (None, 512)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout_1 (Dropout)                  │ (None, 512)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_2 (Dense)                      │ (None, 256)                 │         131,328 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ batch_normalization_2                │ (None, 256)                 │           1,024 │
# │ (BatchNormalization)                 │                             │                 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ activation_2 (Activation)            │ (None, 256)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout_2 (Dropout)                  │ (None, 256)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_3 (Dense)                      │ (None, 128)                 │          32,896 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ batch_normalization_3                │ (None, 128)                 │             512 │
# │ (BatchNormalization)                 │                             │                 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ activation_3 (Activation)            │ (None, 128)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout_3 (Dropout)                  │ (None, 128)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_4 (Dense)                      │ (None, 10)                  │           1,290 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 304,522 (1.16 MB)
#  Trainable params: 302,218 (1.15 MB)
#  Non-trainable params: 2,304 (9.00 KB)
##############
############## Training ###############
# use early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True,verbose=1)
# adjust the learning rate automatically if the training stalls
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,       # reduce LR by half
    patience=5,       # wait 5 epochs before reducing
    min_lr=1e-6,      # minimum learning rate
    verbose=1          # print update when LR is reduced
)
history = model.fit(X_train,y_train,validation_data=(X_test, y_test),epochs=100,batch_size=32,callbacks=[early_stop,reduce_lr],verbose=1)
#Plot accuracy and loss history
util.plot_history_info(history)

############## Evaluation ###############
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
##############
# Test Loss: 0.6131
# Test Accuracy: 0.7911
##############
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=le.classes_))
##############
#                   precision    recall  f1-score   support
#  air_conditioner       0.82      0.93      0.87       200
#         car_horn       0.85      0.66      0.75        86
# children_playing       0.65      0.66      0.66       200
#         dog_bark       0.85      0.64      0.73       200
#         drilling       0.88      0.81      0.84       200
#    engine_idling       0.87      0.94      0.91       200
#         gun_shot       0.77      0.67      0.71        75
#       jackhammer       0.79      0.92      0.85       200
#            siren       0.91      0.93      0.92       186
#     street_music       0.58      0.61      0.60       200

#         accuracy                           0.79      1747
#        macro avg       0.80      0.78      0.78      1747
#     weighted avg       0.79      0.79      0.79      1747
##############
roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
print(f"ROC AUC Score: {roc_auc:.4f}")
##############
# ROC AUC Score: 0.9755
##############
util.plot_confusion_matrix(y_true, y_pred, le)


#----NOTE
#----Same modeling practice has been done using ZCR as the main feature
# zcr_df=df.copy()
# zcr_df.drop(columns=["MFCCs"],axis=1,inplace=True)
#----Evaluation results show lower accuracy and higher loss in model fitting and various miss-classification
