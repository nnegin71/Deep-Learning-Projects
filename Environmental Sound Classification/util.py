# import libraries
import librosa
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


#--------------Feature Extraction
# Mel Frequency Cepstral Coefficient
def mfcc_extractor(file_name):
    audio,sr=librosa.load(file_name)
    #extract mfcc coefficients, return a 2D=(num_coef X frame) matrix
    mfcc=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=10)
    # take the avergae of coefficient to return 1D= (num_coeff) vector 
    mfcc_scaled=np.mean(mfcc.T,axis=0)
    return mfcc_scaled

# Zero Crossing Rate: Determine noisiness of the signals
def zcr_extractor(file_name, frame_size,hop_length ):
    audio,sr=librosa.load(file_name)
    zcr=librosa.feature.zero_crossing_rate(audio,frame_length=frame_size,hop_length=hop_length)[0]
    zcr_scaled=np.mean(zcr.T,axis=0)
    return zcr_scaled


#----------------Evaluation
def plot_history_info(history):
    # Plot accuracy
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./result/Training_history.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, le):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='gnuplot2', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('./result/ConfusionMatrix.png')
    plt.show()