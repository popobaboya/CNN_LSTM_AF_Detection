import tensorflow as tf

# file handling functionality
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
# useful utilities
import time
import pickle

# let's do datascience ...
import numpy as np

# from keras.models import Sequential
from keras import layers

# from keras.optimizers import RMSprop


from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint

# fix random seed for reproducibility
# 재현성을 위한 랜덤 seed 고정, 동일한 방법으로 동일한 결과가 도출됨을 확인하기 위함
seed = 1337
np.random.seed(seed) #결과값 최대한 동일하게 하려고 설정

# tell the application whether we are running on a server or not (so as to
# influence which backend matplotlib uses for saving plots)
# 프로그램에게 서버에서 실행중인지 알림, plots 를 저장할 때 matplotlib 가 영향을 미치도록
headless = False

#
# get the data
# 데이터 가져옴

# load the npz file
# npz 파일 로드 (numpy 배열 파일)
data_path = './data/training_and_validation.npz'
af_data   = np.load(data_path)

# extract the training and validation data sets from this data
# npz 파일에서 훈련 데이터와 검증 데이터 추출
x_train = af_data['x_train']
y_train = af_data['y_train']
x_test  = af_data['x_test']
y_test  = af_data['y_test']

print("GPU를 사용한 학습")

with tf.device("/gpu:0"):

    x_train_tf = tf.convert_to_tensor(x_train, dtype=tf.int32)
    x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.int32)
    y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.int64)
    y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.int64)

    # create and train the model
    # 모델 생성과 훈련

    # set the model parameters
    # 모델 파라미터 설정
    n_timesteps = x_train_tf.shape[1] # 배열의 형태 저장 ex) 2x3 배열의 경우 (2,3) 튜플
    mode = 'concat'  # concat 은 데이터를 연결하는 함수, 해당 구문의 정확한 의미는 모르겠음, 변수가 사용이 안됨
    n_epochs = 80  # epoch 설정 전체 데이터셋을 1회 학습시킨 것이 1epoch
    batch_size = 1024
    # batch size => 한번의 연산에 들어가는 데이터의 크기, 너무 크면 학습 속도 저하, 메모리 부족 문제 발생할 수 있다.
    # 반대로 너무 작으면 적은 데이터로 가중치는 업데이트 하고, 이 업데이트가 자주 발생하므로 훈련이 불안정해진다.

    # create a bidirectional lstm model (based around the model in:
    # https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout
    # )
    # 양방향 lstm 모델 생성, x를 생성하고 생성한 x를 새로운 함수에 넣어서 다시 x에 넣고를 반복


    inp = Input(shape=(n_timesteps,1,))
    # print(inp.shape) 100, 1
    inp = (100,1,)

    model = Sequential()
    # model.add(Input(shape=(100, 1))) 
    model.add(layers.Conv1D(200, 3, activation='relu',
                            input_shape=inp))
    # model.add(MaxPooling1D(10)) 98
    model.add(layers.Conv1D(100, 3, activation='relu'))
    # model.add(layers.LSTM(94, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)) -> 4/30 am12:04
    model.add(layers.LSTM(200, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
    # model.add(MaxPooling1D())
    model.add(GlobalMaxPooling1D())
    # model.add(Dense(50, input_dim=10, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid')) 

    es = tf.keras.callbacks.EarlySt opping(monitor='loss', patience=80)  
            
    # set the optimiser
    # optimizer 설정
    # 딥러닝에서 모델을 설계할 때 파라미터를 최적화 시켜야 좋은 성능을 보여준다.
    # optimizer 는 학습 프로세스에서 파라미터를 갱신시킴으로 파라미터를 최적화 시키는 역할을 한다.
    opt = Adam() # adam 은 optimizer 의 종류 중 하나이다.

    # compile the model
    # 모델 컴파일
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
            
            
    # set up a model checkpoint callback (including making the directory where to 
    # save our weights)
    # 모델 checkpoint callback 세팅 ( + 가중치 저장 디렉토리 설정)
    directory = './model/CNN_LSTM/initial_runs_{0}/'.format(time.strftime("%Y%m%d_%H%M"))
    os.makedirs(directory)
    filename  = 'CNN_LSTM_af_lstm_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpointer = ModelCheckpoint(filepath=directory+filename,
                                verbose=1, 
                                save_best_only=True)
    # modelcheckpoint 는 모델이 학습하면서 정의한 조건을 만족했을 때 가중치를 중간저장 함
    # 중간에 memory overflow 나 crash 가 났을 경우 weight 을 불러와 학습을 이어나갈 수 있기 때문

    # fit the model
    # fit 은 모델 학습 함수이다
    # fit 의 return 값으로 history 객체를 얻을 수 있다.

    history = model.fit(x_train_tf, y_train_tf,  # 훈련데이터, x : 입력 데이터, y: 라벨 값
                        epochs=n_epochs,  # epoch 횟수
                        batch_size=batch_size,  # 기울기를 업데이트 할 샘플의 갯수
                        validation_data=(x_test_tf, y_test_tf),  # 검증 데이터
                        shuffle=True,  # 매 epoch 마다 샘플의 순서를 뒤섞을지 여부
                        callbacks=[checkpointer, es])  # 훈련 진행 중 적용 될 collback 의 리스트          

    # get the best validation accuracy
    best_accuracy = max(history.history['val_acc'])  # history 검증 정확도 기록중에서 가장 높은 값을 가져옴
    print('best validation accuracy = {0:f}'.format(best_accuracy))  # 출력

    # pickle the history so we can use it later
    with open(directory + 'training_history', 'wb') as file:
        pickle.dump(history.history, file)  # pickle 은 문자열이나 값을 한 번에 전달, write 대신 사용하는 거라고 보면 될 듯

    # set matplotlib to use a backend that doesn't need a display if we are 
    # running remotely
    # 서버 사용 시
    if headless:
        import matplotlib
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    # plot the results
    # 결과에 대한 plot 생성

    # accuracy
    # 정확도
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)  # 111 은 plot 의 행의 수, 열의 수 인덱스 각각을 의미
    plt.plot(history.history['acc'])  # history 함수는 keras 의 기능 중 하나로 학습 이력을 return 한다. acc 는 훈련 정확도
    plt.plot(history.history['val_acc'])  # val_acc 는 훈련 정확도
    plt.title('training and validation accuracy of af diagnosis')  # plot title 설정
    plt.ylabel('accuracy')  # y축 라벨을 accuracy 로
    plt.xlabel('epoch')  # x 축 라벨을 epoch 로
    plt.legend(['train', 'test'], loc='upper left')  # legend 함수는 범례를 표시한다. train 과 test 를 범례로 하고 loc 은 위치 설정
    plt.text(0.4, 0.05, 
            ('validation accuracy = {0:.3f}'.format(best_accuracy)), 
            ha='left', va='center', 
            transform=ax1.transAxes)  # text 는 텍스트 삽입 함수
    plt.savefig('CNN_LSTM_training_accuracy_{0}.png'
                .format(time.strftime("%Y%m%d_%H%M")))  # png 파일로 저장


    # loss
    # 손실값
    f2 = plt.figure()
    ax2 = f2.add_subplot(111)  # 111 은 plot 의 행의 수, 열의 수 인덱스 각각을 의미
    plt.plot(history.history['loss'])  # history 함수는 keras 의 기능 중 하나로 학습 이력을 return 한다. loss 는 훈련 손실값
    plt.plot(history.history['val_loss'])  # val_loss 는 검증 손실값
    plt.title('training and validation loss of af diagnosis')  # plot 제목
    plt.ylabel('loss')  # plot y 축
    plt.xlabel('epoch')  # plot x 축
    plt.legend(['train', 'test'], loc='upper right')  # 범례 설정
    plt.text(0.4, 0.05,
            ('validation loss = {0:.3f}'
            .format(min(history.history['val_loss']))), 
            ha='right', va='top', 
            transform=ax2.transAxes)  # text 삽입
    plt.savefig('CNN_LSTM_training_loss_{0}.png'
                .format(time.strftime("%Y%m%d_%H%M")))  # png 로 저장

    # we're all done!
    # 끝
    print('all done!')