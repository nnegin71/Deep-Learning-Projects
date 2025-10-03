#----------import libraries
import numpy as np 

import librosa
import librosa.display

from IPython import display
from IPython.display import Audio

import matplotlib.pyplot as plt
import seaborn as sns

def explore_audio(audio_file):
    """
    This function explores a single audio file
    input: audio waveform
    output: audio visualization including time-series and multipe spectrograms
    """
    print("Play the audio file")
    Audio(audio_file)

    # get time-series and sampling rate
    y,sr = librosa.load(audio_file)

    # setup subplot for visualization
    plt.figure(figsize=(15,7))

    # Time-series waveform
    plt.subplot(3,2,1)
    librosa.display.waveshow(y=y , sr= sr , color='b')
    plt.title('Waveform')
    plt.xlabel('Time(sec)')
    plt.ylabel('Amplitude')

    # Linear Spectrogram
    plt.subplot(3,2,2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr= sr , x_axis = 'time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')

    # Logarithmic Spectrogram
    plt.subplot(3,2,3)
    librosa.display.specshow(D, sr= sr , x_axis = 'time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Logarithmic Spectrogram')

    # Mel Spectrogram
    plt.subplot(3,2,4)
    D_mel = librosa.feature.melspectrogram(y= y, sr=sr , n_mels= 256)
    s_db_mel = librosa.amplitude_to_db(D_mel, ref=np.max)
    librosa.display.specshow(s_db_mel, sr= sr , x_axis = 'time',y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')

    # Mel Frequency Cepstral Coefficient (MFCC)
    plt.subplot(3,2,5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, sr= sr , x_axis = 'time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Frequency Cepstral Coefficient (MFCC)')

    plt.tight_layout()
    plt.savefig('./result/explore_audio_example.png')
    plt.show()


def explore_data_distribution(df):
    """
    This function explores data distribution based on audio file duration, different class type 
    input: a dataframe, which includes following info
    #       slice_file_name    fsID  start        end  salience  fold  classID             class  duration
    # 0    100032-3-0-0.wav  100032    0.0   0.317551         1     5        3          dog_bark  0.317551
    # 1  100263-2-0-117.wav  100263   58.5  62.500000         1     5        2  children_playing  4.000000
    # 2  100263-2-0-121.wav  100263   60.5  64.500000         1     5        2  children_playing  4.000000
    # 3  100263-2-0-126.wav  100263   63.0  67.000000         1     5        2  children_playing  4.000000
    # 4  100263-2-0-137.wav  100263   68.5  72.500000         1     5        2  children_playing  4.000000
    output: visualization of data distribution
    """
    # add duration column to the data from 
    df['duration'] = df['end']-df['start']
    ##############
    #       slice_file_name    fsID  start        end  salience  fold  classID             class  duration
    # 0    100032-3-0-0.wav  100032    0.0   0.317551         1     5        3          dog_bark  0.317551
    # 1  100263-2-0-117.wav  100263   58.5  62.500000         1     5        2  children_playing  4.000000
    # 2  100263-2-0-121.wav  100263   60.5  64.500000         1     5        2  children_playing  4.000000
    # 3  100263-2-0-126.wav  100263   63.0  67.000000         1     5        2  children_playing  4.000000
    # 4  100263-2-0-137.wav  100263   68.5  72.500000         1     5        2  children_playing  4.000000
    ##############
    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 12))

    # 1. Class Distribution
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='class')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # 2. Duration
    plt.subplot(2, 2, 2)
    sns.barplot(data=df[: 10], x='slice_file_name', y='duration')
    plt.title('Duration per File')
    plt.xlabel('File Name')
    plt.ylabel('Duration (s)')
    plt.xticks(rotation=45)

    # 3. Fold Distribution
    plt.subplot(2, 2, 3)
    sns.countplot(data=df, x='fold')
    plt.title('Fold Distribution')
    plt.xlabel('Fold')
    plt.ylabel('Count')

    # 4. ClassID vs. Duration
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='classID', y='duration', hue='class', size='duration', sizes=(50, 200))
    plt.title('ClassID vs. Duration')
    plt.xlabel('Class ID')
    plt.ylabel('Duration (s)')
    plt.legend(title='Class')

    plt.tight_layout()
    plt.savefig('./result/explore_data_distribution.png')
    plt.show()

    #drop duration column
    df.drop(columns=["duration"],axis=1,inplace=True)


    



