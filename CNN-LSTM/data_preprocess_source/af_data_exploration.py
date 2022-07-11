"""
af_data_exploration.py
exploring the atrial fibrillation data set
author:     alex shenfield
date:       01/04/2018
"""

# file handling functionality
import os
import glob

# let's do datascience ...
import pandas as pd
import numpy as np

# fix random seed for reproduciblity
# 랜덤성을 제어하기 위함, 난수의 생성 패턴을 동일하게 관리
seed = 1337
np.random.seed(seed)

#
# read and save the data to use in training
#

# set the directory where the data lives
# 데이터셋이 있는 디렉토리 경로 설정
root_dir = ('./data/' +
            'patient_data_100_beat_window_99_beat_overlap/train/')

# get the patient IDs
filelist = glob.glob(root_dir + '*.csv', recursive=True)
patients  = [(os.path.split(i)[1]).split('_')[0] for i in filelist]
print(patients)
# read all the data into a single data frame
# 모든 데이터를 싱글 데이터 프레임으로 읽어들임
frames = [pd.read_csv(p, header=None) for p in filelist]
data   = pd.concat(frames, ignore_index=True)


# show what our data looks like
# 총 데이터 개수와, 환자 수(csv 파일 수)
print('we have {0} data points from {1} patients!'.format(data.shape,
      len(patients)))


# split the data into variables and targets (0 = no af, 1 = af)
# 데이터를 분류하여 변수에 저장
x_data = data.iloc[:,1:].values  # 모든 행에 대해 1번 행 이후 데이터
print(x_data)
y_data = data.iloc[:,0].values  # 모든 행에 대해 0번 열

'''
list1 = []
list2 = []
temp = 0
for y in y_data:
    if y > 0:
        list1.append(temp)
    temp = temp + 1

    if temp > 43905:
        break;
'''

y_data[y_data > 0] = 1  # y_data 에서 값이 0 보다 큰 경우 1로 저장(af)

i = 0
j = 0
for x in y_data:
    if x == 1:
        i = i+1
    elif x == 0:
        j = j+1
print('i = ', i, 'j = ', j)

'''
temp = 0
for y in y_data:
    if y > 0:
        list2.append(temp)
    temp = temp + 1

    if temp > 43905:
        break;

print(len(list1), len(list2))

for i in range(len(list1)):
    print(list1[i], list2[i])
'''

# save the data with numpy so we can use it later
# 나중에 사용하기 위한 numpy 배열을 파일로 저장
datafile = './data/training_data.npz'
np.savez(datafile, 
         x_data=x_data, y_data=y_data)

# count the number of samples exhibitning af
# 전체 데이터중 심방세동을 포함한 데이터의 개수 카운트
print('there are {0} out of {1} samples that have at least one beat that '
      'is classified as atrial fibrillation'.format(sum(y_data), len(y_data)))

#
# lets visualise some of the data 
# 몇가지 데이터들을 시각화, plot 으로 만들어 png 파일로 저장

# import matplotlib plotting magic
import matplotlib.pyplot as plt

# lets plot every patient in our training and validation sets
i = 0
for patient in frames:

    # reshape the data sequences of a patient until we have a single 
    # consecutive heart rate trace (like the original data)
    patient.drop(patient.columns[0], axis=1, inplace=True)
    hr_trace = patient.iloc[::100, :]
    hr_trace_seq = hr_trace.values.reshape(1,-1)
    plt.figure()
    plt.title('patient id = {0}'.format(patients[i]))
    plt.plot(hr_trace_seq.transpose(), '.')
    plt.savefig('./log/figures/4_fixed_data/{0}.png'.format(patients[i]))
    i = i + 1

#
# split the data into training (75%) and validation (25%) to use in our 
# initial model development
# 데이터의 4분의 3은 훈련용, 4분의 1은 테스트용으로 split
# 10-fold 검증을 위한 것 같음
# note: when we start to properly fine tune the models we should use 10-fold 
# cross validation to evaluate the effects of the model structure and
# parameters 
#

# create train and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    stratify=y_data, 
                                                    test_size=0.25,
                                                    random_state=seed)

# reformat the training and test inputs to be in the format that the lstm 
# wants
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# save the data with numpy so we can use it later
# npz 파일로 저장
datafile = './data/training_and_validation.npz'
np.savez(datafile, 
         x_train=x_train, x_test=x_test, 
         y_train=y_train, y_test=y_test)