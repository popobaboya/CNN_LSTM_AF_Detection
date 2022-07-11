# CNN_LSTM_AF_Detection
* 본 프로젝트는 CNN-LSTM 모델을 사용하여 심방세동을 감지하기 위한 프로젝트이다.
* 모델의 학습 및 테스트 데이터로 RR interval 만을 사용하였으며, 따라서 ECG 와 PPG 데이터 모두에서 사용이 가능하다.
* 학습 및 테스트 데이터는 MIT-BIH Atrial Fibrillation Database 의 23명의 환자의 ECG 데이터를 사용하였다.

## CNN-LSTM Deep Learning Model
![image](https://user-images.githubusercontent.com/61726550/178174695-39b9f82d-d2c5-49f5-8b07-404daf2ae30d.png)

본 프로젝트에서 구현된 모델은 2개의 1D convolution neural network(CNN) 층들과 1개의 long short-term memory(LSTM) 층의 조합으로 이루어져 있다. 2개의 1D CNN 층의 각각의 filter의 개수는 200과 100으로 설정하였고 kernel 크기는 동일하게 3으로 설정하였다. 단일 LSTM 층의 셀 수는 입력 시퀀스 길이의 2배로 설정되었으며 최종 분류를 수행하는 상위 모델로 사용되었다. 1차원의 Global max pooling은 단방향 LSTM 층과 완전 연결 층들 사이에 사용되어 2개의 convolution layer 층과 단방향 LSTM 층에 의해 생성된 출력 시퀀스들의 특징들을 압축했다. CNN-LSTM 층은 효과적으로 input HR(Heart rate) 데이터 시퀀스의 특징들을 학습하고 추출한 후, 이러한 특징들을 완전 연결 계층에 전달하여 심방세동 증상이 있는지 여부를 분류한다.


## Tizen - Android Applicaiton
![image](https://user-images.githubusercontent.com/61726550/178177206-2b6dd339-0a3a-4e1b-badf-ac10591b5b93.png)

설계된 CNN-LSTM 딥러닝 모델을 사용하여 심방세동을 검출하는 서비스의 시스템을 설계하였다. 웨어러블 기기에서 RR interval 과 HRM 데이터를 안드로이드 기기로 전송하면 안드로이드 앱에서는 해당 데이터를 CNN-LSTM 모델을 사용하여 심방세동 유무를 판별한다. 안드로이드 앱에서는 검출 결과를 출력하며, 심방세동 발생시 또는 심박이 일정 시간 이상 멈출 시 보호자에게 연락을 취한다.

현재는 갤럭시 워치3 에서 RR interval 과 HR 데이터를 자바 안드로이드 앱에 전송하는 것까지 구현한 상태이며, CNN-LSTM 모델로 테스트하여 결과를 표출하는 과정은 구현되지 않은 상태이다. 

## Documentation
* 자세한 내용은 CNN_LSTM_AF_Detection_Documentation.pdf 파일을 참고
https://github.com/popobaboya/CNN_LSTM_AF_Detection/blob/main/CNN_LSTM_AF_Deteciton_Documentation.pdf

## Collaborator
한동대학교 전산전자공학부
* 박천성 (Developer)
* 노은호 (Developer)
* 안민규 (Professor)

## References
* MIT-BIH Atrial Fibrillation Database : https://physionet.org/content/afdb/1.0.0/
* PhysioBank ATM : https://archive.physionet.org/cgi-bin/atm/ATM
* Tizen Developers : https://developer.tizen.org/?langredirect=1
* Wfdb-python : https://github.com/MIT-LCP/wfdb-python
* Rnn-based-af-detection : https://github.com/al3xsh/rnn-based-af-detection
