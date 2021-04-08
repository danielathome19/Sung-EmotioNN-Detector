# IMPORT NECESSARY LIBRARIES
import glob
import math
import time
from threading import Thread
import librosa
import matplotlib.pyplot as plt
import librosa.display
import pyaudio
import wave
from IPython.display import Audio
import numpy as np
from matplotlib.pyplot import specgram
import pandas as pd
from sklearn.metrics import confusion_matrix
import IPython.display as ipd  # To play sound in the notebook
import os  # interface with underlying OS that python is running on
import soundfile as sf
import sys
import warnings
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization, Dense
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from sklearn import tree
from sklearn.dummy import DummyClassifier
from keras.utils import np_utils, to_categorical
from pydub import AudioSegment
# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


class SplitWavAudio:
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        # self.filepath = folder + '\\' + filename
        self.audio = AudioSegment.from_wav(self.filename)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_sec, to_sec, split_filename):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '/' + split_filename[split_filename.index("/")+1:], format="wav")

    def multiple_split(self, sec_per_split):
        total_sec = math.ceil(self.get_duration())
        for i in range(0, total_sec, sec_per_split):
            split_fn = self.filename[:self.filename.index('.')] + '_' + str(i) + '.wav'
            self.single_split(i, i + sec_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_sec - sec_per_split:
                print('All splits completed successfully')


def demo(audiofilepath='Audio_Song_Actors_01-24/Actor_01/03-02-01-01-01-01-01.wav',
         wavtitle='Waveplot - Male Neutral', wavfigtitle='Waveplot_MaleNeutral.png',
         outputpath='AudioFiles/MaleNeutral.wav',
         title='Mel Spectrogram - Male Neutral', figfilename='MelSpec_MaleNeutral.png'):
    # LOAD IN FILE
    x, sr = librosa.load(audiofilepath)
    # DISPLAY WAVEPLOT
    plt.figure(figsize=(8, 4))
    librosa.display.waveplot(x, sr=sr)
    plt.title(wavtitle)
    plt.savefig(wavfigtitle)
    # PLAY AUDIO FILE
    sf.write(outputpath, x, sr)
    Audio(data=x, rate=sr)
    # CREATE LOG MEL SPECTROGRAM
    spectrogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128, fmax=8000)
    spectrogram = librosa.power_to_db(spectrogram)

    librosa.display.specshow(spectrogram, y_axis='mel', fmax=8000, x_axis='time')
    plt.title(title)
    plt.savefig(figfilename)
    plt.colorbar(format='%+2.0f dB')


def make_classifier(X_train, lrval=0.0001):
    # BUILD CNN MODEL
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(64, kernel_size=10, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(layers.Conv1D(128, kernel_size=10, activation='relu', kernel_regularizer=l2(0.01),
                            bias_regularizer=l2(0.01)))
    model.add(layers.MaxPooling1D(pool_size=6))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv1D(128, kernel_size=10, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=6))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(6, activation='sigmoid'))
    opt = keras.optimizers.Adam(lr=lrval)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def train():
    """ PREPROCESSING """
    # CREATE DIRECTORY OF AUDIO FILES
    audio = "Audio_Song_Actors_01-24/"
    actor_folders = os.listdir(audio)  # list files in audio directory
    actor_folders.sort()
    print(actor_folders[0:5])
    # CREATE FUNCTION TO EXTRACT EMOTION NUMBER, ACTOR AND GENDER LABEL
    emotion = []
    gender = []
    actor = []
    file_path = []
    for i in actor_folders:
        filename = os.listdir(audio + i)  # iterate over Actor folders
        for f in filename:  # go through files in Actor folder
            part = f.split('.')[0].split('-')
            emotion.append(int(part[2]))
            actor.append(int(part[6]))
            bg = int(part[6])
            if bg % 2 == 0:
                bg = "female"
            else:
                bg = "male"
            gender.append(bg)
            file_path.append(audio + i + '/' + f)

    # PUT EXTRACTED LABELS WITH FILEPATH INTO DATAFRAME
    audio_df = pd.DataFrame(emotion)
    audio_df = audio_df.replace(
        {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'})
    audio_df = pd.concat([pd.DataFrame(gender), audio_df, pd.DataFrame(actor)], axis=1)
    audio_df.columns = ['gender', 'emotion', 'actor']
    audio_df = pd.concat([audio_df, pd.DataFrame(file_path, columns=['path'])], axis=1)
    print(audio_df)
    # ENSURE GENDER,EMOTION, AND ACTOR COLUMN VALUES ARE CORRECT
    pd.set_option('display.max_colwidth', -1)
    audio_df.sample(10)
    # LOOK AT DISTRIBUTION OF CLASSES
    audio_df.emotion.value_counts().plot(kind='bar')
    # EXPORT TO CSV
    audio_df.to_csv('Uploads/audio.csv')

    """ EXTRACT FEATURES """
    # ITERATE OVER ALL AUDIO FILES AND EXTRACT LOG MEL SPECTROGRAM MEAN VALUES INTO DF FOR MODELING
    df = pd.DataFrame(columns=['mel_spectrogram'])

    counter = 0

    for index, path in enumerate(audio_df.path):
        X, sample_rate = librosa.load(path, res_type='kaiser_fast', duration=3, sr=44100, offset=0.5)

        # get the mel-scaled spectrogram (ransform both the y-axis (frequency) to log scale, and the
        # “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.)
        spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128, fmax=8000)
        db_spec = librosa.power_to_db(spectrogram)
        # temporally average spectrogram
        log_spectrogram = np.mean(db_spec, axis=0)

        # Mel-frequency cepstral coefficients (MFCCs)
        #     mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
        #     mfcc=np.mean(mfcc,axis=0)

        # compute chroma energy (pertains to 12 different pitch classes)
        #     chroma = librosa.feature.chroma_stft(y=X, sr=sample_rate)
        #     chroma = np.mean(chroma, axis = 0)

        # compute spectral contrast
        #     contrast = librosa.feature.spectral_contrast(y=X, sr=sample_rate)
        #     contrast = np.mean(contrast, axis= 0)

        # compute zero-crossing-rate (zcr:the zcr is the rate of sign changes along a signal i.e.m the rate at
        #     which the signal changes from positive to negative or back - separation of voiced andunvoiced speech.)
        #     zcr = librosa.feature.zero_crossing_rate(y=X)
        #     zcr = np.mean(zcr, axis= 0)

        df.loc[counter] = [log_spectrogram]
        counter = counter + 1

    print(len(df))
    df.head()
    # TURN ARRAY INTO LIST AND JOIN WITH AUDIO_DF TO GET CORRESPONDING EMOTION LABELS
    df_combined = pd.concat([audio_df, pd.DataFrame(df['mel_spectrogram'].values.tolist())], axis=1)
    df_combined = df_combined.fillna(0)
    # DROP PATH COLUMN FOR MODELING
    df_combined.drop(columns='path', inplace=True)
    # CHECK TOP 5 ROWS
    df_combined.head()
    # TRAIN TEST SPLIT DATA
    train, test = train_test_split(df_combined, test_size=0.2, random_state=0,
                                   stratify=df_combined[['emotion', 'gender', 'actor']])
    X_train = train.iloc[:, 3:]
    y_train = train.iloc[:, :2].drop(columns=['gender'])
    print(X_train.shape)
    X_test = test.iloc[:, 3:]
    y_test = test.iloc[:, :2].drop(columns=['gender'])
    print(X_test.shape)
    # NORMALIZE DATA
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    # TURN DATA INTO ARRAYS FOR KERAS
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    # ONE HOT ENCODE THE TARGET
    # CNN REQUIRES INPUT AND OUTPUT ARE NUMBERS
    lb = LabelEncoder()
    y_train = to_categorical(lb.fit_transform(y_train))
    y_test = to_categorical(lb.fit_transform(y_test))
    # np.save('classes.npy', lb.classes_)
    # return

    print(y_test[0:10])
    print(lb.classes_)

    print(X_train.shape)
    print(X_test.shape)

    """ BASE MODEL """
    dummy_clf = DummyClassifier(strategy="stratified")
    dummy_clf.fit(X_train, y_train)
    DummyClassifier(strategy='stratified')
    dummy_clf.predict(X_test)
    dummy_clf.score(X_test, y_test)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    clf.predict(X_test)
    clf.score(X_test, y_test)

    """ INITIAL MODEL """
    # RESHAPE DATA TO INCLUDE 3D TENSOR
    X_train = X_train[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]
    print(X_train)
    # BUILD 1D CNN LAYERS
    model = make_classifier(X_train, 0.001)
    model.summary()
    if not os.path.isfile('Model_Diagram.png'):  # keras plot_model library
        plot_model(model, to_file='Model_Diagram.png', show_shapes=True, show_layer_names=True)
    # FIT MODEL AND USE CHECKPOINT TO SAVE BEST MODEL
    checkpoint = ModelCheckpoint("best_initial_model.hdf5", monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max', save_freq='epoch', save_weights_only=True)

    model_history = model.fit(X_train, y_train, batch_size=32, epochs=2000, validation_data=(X_test, y_test),
                              callbacks=[checkpoint])

    plt.close()
    # PLOT MODEL HISTORY OF ACCURACY AND LOSS OVER EPOCHS
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Initial_Model_Accuracy.png')
    plt.show()
    # pd.DataFrame(model_history.history).plot()  # figsize=(8, 5)
    # plt.show()

    # summarize history for loss
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Initial_Model_loss.png')
    plt.show()

    """ POST-MODEL ANALYSIS """

    # PRINT LOSS AND ACCURACY PERCENTAGE ON TEST SET
    print("Loss of the model is - ", model.evaluate(X_test, y_test)[0])
    print("Accuracy of the model is - ", model.evaluate(X_test, y_test)[1] * 100, "%")
    # PREDICTIONS
    predictions = model.predict(X_test)
    predictions = predictions.argmax(axis=1)
    predictions = predictions.astype(int).flatten()
    predictions = (lb.inverse_transform(predictions))
    predictions = pd.DataFrame({'Predicted Values': predictions})

    # ACTUAL LABELS
    actual = y_test.argmax(axis=1)
    actual = actual.astype(int).flatten()
    actual = (lb.inverse_transform(actual))
    actual = pd.DataFrame({'Actual Values': actual})

    # COMBINE BOTH
    finaldf = actual.join(predictions)
    print(finaldf[140:150])
    # CREATE CONFUSION MATRIX OF ACTUAL VS. PREDICTION
    cm = confusion_matrix(actual, predictions)
    plt.figure(figsize=(12, 10))
    cm = pd.DataFrame(cm, index=[i for i in lb.classes_], columns=[i for i in lb.classes_])
    ax = sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('Confusion Matrix', size=20)
    plt.xlabel('Predicted Labels', size=14)
    plt.ylabel('Actual Labels', size=14)
    plt.savefig('Initial_Model_Confusion_Matrix.png')
    plt.show()
    print(classification_report(actual, predictions,
                                target_names=['angry', 'calm', 'fear', 'happy', 'neutral', 'sad']))

    """ HYPERPARAMETER TUNING """
    # TRAIN TEST SPLIT DATA
    train, test = train_test_split(df_combined, test_size=0.2, random_state=0,
                                   stratify=df_combined[['gender', 'actor']])

    X_train = train.iloc[:, 3:]
    y_train = train.iloc[:, :2].drop(columns=['gender'])
    print(X_train.shape)

    X_test = test.iloc[:, 3:]
    y_test = test.iloc[:, :2].drop(columns=['gender'])
    print(X_test.shape)

    # NORMALIZE DATA
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # TURN DATA INTO ARRAYS FOR KERAS
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # RESHAPE TO INCLUDE 3D TENSOR
    X_train = X_train[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]

    lb = LabelEncoder()

    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    y_trainHot = np.argmax(y_train, axis=1)
    # GRID SEARCH PARAMETERS TO FIND BEST VALUES
    classifier = KerasClassifier(build_fn=make_classifier(X_train))
    params = {
        'batch_size': [30, 32, 34],
        'nb_epoch': [25, 50, 75],
        'optimizer': ['adam', 'SGD']}

    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=params,
                               scoring='accuracy',
                               cv=5)

    grid_search = grid_search.fit(X_train, y_trainHot)
    print(grid_search.best_params_)
    print(grid_search.best_score_)


def predict(audio_folder='TestFiles/', getOnlyLast=False, waitForFileName='', verbosePrint=True):
    audio = audio_folder
    # CREATE FUNCTION TO EXTRACT EMOTION NUMBER, ACTOR AND GENDER LABEL
    file_path = []
    if getOnlyLast:
        while not os.path.exists(waitForFileName):
            time.sleep(1/1000)
        list_of_files = glob.glob(f'{audio_folder}*')  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        file_path.append(latest_file)
    else:
        for i in os.listdir(audio):
            file_path.append(audio + i)

    audio_df = pd.DataFrame(file_path, columns=['path'])
    print(audio_df)

    """ EXTRACT FEATURES """
    # ITERATE OVER ALL AUDIO FILES AND EXTRACT LOG MEL SPECTROGRAM MEAN VALUES INTO DF FOR MODELING
    df = pd.DataFrame(columns=['mel_spectrogram'])

    counter = 0

    for index, path in enumerate(audio_df.path):
        X, sample_rate = librosa.load(path, res_type='kaiser_fast', duration=3, sr=44100, offset=0.5)

        # get the mel-scaled spectrogram (ransform both the y-axis (frequency) to log scale, and the
        # “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.)
        spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128, fmax=8000)
        db_spec = librosa.power_to_db(spectrogram)
        # temporally average spectrogram
        log_spectrogram = np.mean(db_spec, axis=0)

        # TODO: """ TODO """
        # Mel-frequency cepstral coefficients (MFCCs)
        #     mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
        #     mfcc=np.mean(mfcc,axis=0)

        # compute chroma energy (pertains to 12 different pitch classes)
        #     chroma = librosa.feature.chroma_stft(y=X, sr=sample_rate)
        #     chroma = np.mean(chroma, axis = 0)

        # compute spectral contrast
        #     contrast = librosa.feature.spectral_contrast(y=X, sr=sample_rate)
        #     contrast = np.mean(contrast, axis= 0)

        # compute zero-crossing-rate (zcr:the zcr is the rate of sign changes along a signal i.e.m the rate at
        #     which the signal changes from positive to negative or back - separation of voiced andunvoiced speech.)
        #     zcr = librosa.feature.zero_crossing_rate(y=X)
        #     zcr = np.mean(zcr, axis= 0)

        df.loc[counter] = [log_spectrogram]
        counter = counter + 1

    print(len(df))
    df.head()
    # TURN ARRAY INTO LIST AND JOIN WITH AUDIO_DF TO GET CORRESPONDING EMOTION LABELS
    df_combined = pd.concat([audio_df, pd.DataFrame(df['mel_spectrogram'].values.tolist())], axis=1)
    df_combined = df_combined.fillna(0)
    # DROP PATH COLUMN FOR MODELING
    df_combined.drop(columns='path', inplace=True)
    # CHECK TOP 5 ROWS
    df_combined.head()
    # TRAIN TEST SPLIT DATA

    # NORMALIZE DATA
    mean = np.mean(df_combined, axis=0)
    std = np.std(df_combined, axis=0)
    X_train = (df_combined - mean) / std
    # TURN DATA INTO ARRAYS FOR KERAS
    X_train = np.array(X_train)
    # ONE HOT ENCODE THE TARGET
    # CNN REQUIRES INPUT AND OUTPUT ARE NUMBERS
    lb = LabelEncoder()
    lb.classes_ = np.load('classes.npy', allow_pickle=True)
    # y_train = to_categorical(lb.fit_transform(y_train))
    # y_test = to_categorical(lb.fit_transform(y_test))
    X_train = X_train[:, :, np.newaxis]

    # print(X_train)
    # print((model.predict(X_train) > 0.5).astype("int32"))
    model = make_classifier(X_train, 0.001)
    model.load_weights('best_initial_model.hdf5')
    if verbosePrint:
        model.summary()
    predictions = model.predict(X_train, batch_size=32)
    predictions = predictions.argmax(axis=1)
    predictions = predictions.astype(int).flatten()
    predictions = (lb.inverse_transform(predictions))
    predictions = pd.DataFrame({'Predicted Values': predictions})
    print(predictions)
    # predictions = model.predict_classes(X_train, verbose=1)
    # print(predictions)


def record(SPLIT_LEN=5, recordingFolder='RecordedFiles/', DURATION: float = -1, realTimePredicting=False):
    CHUNK = 2 ** 5
    CHANNELS = 1
    RATE = 48000
    FORMAT = pyaudio.paInt16
    LEN = SPLIT_LEN  # seconds

    def record_audio():
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        # player = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

        print("* recording")
        frames = []

        for i in range(int(LEN * RATE / CHUNK)):  # go for a LEN seconds
            data = stream.read(CHUNK)
            frames.append(data)
            # data = np.fromstring(stream.read(CHUNK), dtype=np.int16)
            # player.write(data, CHUNK)

        print("* done recording")
        stream.stop_stream()
        stream.close()
        p.terminate()

        number_files = len(os.listdir(recordingFolder))
        output_wav_filename = recordingFolder + f'MyTest{number_files + 1}.wav'
        wf = wave.open(output_wav_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        return output_wav_filename

    if DURATION == -1:
        record_audio()
    else:
        timecnt: float = 0.0
        while float(timecnt / 60.0) != float(DURATION):
            outputfn = record_audio()
            if realTimePredicting:
                thr = Thread(target=predict(recordingFolder, getOnlyLast=True, waitForFileName=outputfn, verbosePrint=False))
                thr.start()
            timecnt += float(LEN)


def realtime(SPLIT_LEN=5, recordingFolder='RealtimeTest/', DURATION: float = -1):
    record(SPLIT_LEN=SPLIT_LEN, recordingFolder=recordingFolder, DURATION=DURATION, realTimePredicting=True)


def separatevocals(audiofilepath, newfilename='new-audio'):
    y, sr = librosa.load(audiofilepath)  # , duration=120) to limit the number of seconds loaded
    S_full, phase = librosa.magphase(librosa.stft(y))

    # Plot a 5-second slice of the spectrum
    idx = slice(*librosa.time_to_frames([30, 35], sr=sr))
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    plt.colorbar()
    plt.tight_layout()
    # The wiggly lines above are due to the vocal component.
    # Our goal is to separate them from the accompanying instrumentation.

    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))

    S_filter = np.minimum(S_full, S_filter)
    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    # Plot the same slice, but separated into its foreground and background
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                             y_axis='log', sr=sr)
    plt.title('Full spectrum')
    plt.colorbar()

    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
                             y_axis='log', sr=sr)
    plt.title('Background')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    plt.title('Foreground')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    new_y = librosa.istft(S_foreground * phase)
    sf.write(f'SongTest/{newfilename}.wav', new_y, sr)


def wavsplit(inputfilepath, splitfolder, SPLIT_LEN=20):
    split_wav = SplitWavAudio(splitfolder, inputfilepath)
    split_wav.multiple_split(sec_per_split=SPLIT_LEN)


if __name__ == '__main__':
    print("Hello world!")
    # demo()
    # demo('TestFiles/03-02-04-01-01-01-01.wav', wavtitle='Wavtest 1', wavfigtitle='Wavtest 1.png',
    #     outputpath='AudioFiles/Test1.wav', title='Mel Spectrogram - Test 1', figfilename='MelSpec - Test1.png')
    # train()
    # predict()
    # record(recordingFolder='DurationTest/', DURATION=0.5)
    # predict('DurationTest/')
    # realtime(SPLIT_LEN=5, DURATION=0.5)
    # separatevocals('SongTest/Dietrich Fischer-Dieskau Allerseelen Richard Strauss.wav')
    # separatevocals('SongTest/Lenskis Aria.wav', 'new-audioLenski')
    # separatevocals('SongTest/saLauridsen.wav', 'naLauridsen')
    separatevocals('SongTest/CoriolanOverture.wav', 'naCorOverture')
    # wavsplit('SongTest/new-audio.wav', 'SongTest/Allerseelen/')
    # wavsplit('SongTest/new-audioLenski.wav', 'SongTest/LenskisAria/')
    # wavsplit('SongTest/naLenski.wav', 'SongTest/LenskisAriaTen/', SPLIT_LEN=10)
    # wavsplit('SongTest/naLenski.wav', 'SongTest/LenskisAriaForty/', SPLIT_LEN=40)
    # wavsplit('SongTest/naLenski.wav', 'SongTest/LenskisAriaSixty/', SPLIT_LEN=60)
    # wavsplit('SongTest/naLenski.wav', 'SongTest/LenskisAriaOneTwenty/', SPLIT_LEN=120)
    # wavsplit('SongTest/Dietrich Fischer-Dieskau Allerseelen Richard Strauss.wav', 'SongTest/AllerseelenUnsplit/')
    # wavsplit('SongTest/Lenskis Aria.wav', 'SongTest/LenskisAriaUnsplit/')
    # wavsplit('SongTest/Lenskis Aria.wav', 'SongTest/LenskisAriaUnsplitTen/', SPLIT_LEN=10)
    # wavsplit('SongTest/naLauridsen.wav', 'SongTest/Lauridsen/')
    # wavsplit('SongTest/saLauridsen.wav', 'SongTest/LauridsenUnsplit/')
    # wavsplit('SongTest/CoriolanOverture.wav', 'SongTest/CoriolanOverture/')
    # wavsplit('SongTest/CoriolanOverture.wav', 'SongTest/CorOvertureForty/', SPLIT_LEN=40)
    # predict('SongTest/Allerseelen/')
    # predict('SongTest/LenskisAria/')
    # predict('SongTest/LenskisAriaTen/')
    # predict('SongTest/LenskisAriaForty/')
    # predict('SongTest/LenskisAriaSixty/')
    # predict('SongTest/LenskisAriaOneTwenty/')
    # predict('SongTest/AllerseelenUnsplit/')
    # predict('SongTest/LenskisAriaUnsplit/')
    # predict('SongTest/LenskisAriaUnsplitTen/')
    # predict('SongTest/LauridsenUnsplit/')
    # predict('SongTest/Lauridsen/')
    # predict('SongTest/CoriolanOverture/')
    # predict('SongTest/CorOvertureForty/')
    # realtime(5, DURATION=.5)
    # predict('RealTimeTest/')
