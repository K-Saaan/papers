## Analysis and Recognition of Depressive Emotion through NLP and Machine Learning

### Problem
현대 사회가 복잠해짐에 따라 우울증이 문제로 대두되고 있다. SNS상 우울증이 의심되는 사용자를 조기에 발견하고, 상담 및 치료현장으로 이끌어 우울증을 예방하고 자살 징후에 대비하기 위함.

### Motivation
우울증이 있거나 의심되는 사람들을 발견해 예방하고자 함.

### Method
- RNN : 84.7% (Accuracy)
- LSTM : 89.6% (Accuracy)
- GRU : 92.2% (Accuracy)

### Experiment
#### Dataset
데이터 수집 플랫폼 : 트위터
우울함과 그렇지 않은 데이터 수집을 위해 각각에 대한 검색 키워드를 설정하고 트위터에서 추출.
트위터에서 GetOldTweet3 package를 사용해 2019년 3월 ~ 2019년 6월 데이터 수집.

<img width="300" alt="표1 슬픔과 기쁨 범주 단어" src="https://github.com/kimdaehyuun/Quanters/assets/111870436/3372bf8b-a093-4c9e-9333-059337ab3c66">

<img width="300" alt="표2 종결어미를 제외한 단어" src="https://github.com/kimdaehyuun/Quanters/assets/111870436/37f7fcde-4ed4-49d6-9372-b468b803c294">

수집 결과
- 우울한 문장 : 77,064
- 우울하지 않은 문장 8,563

관련성이 떨어지는 문장 제거를 위해 http링크와 타 SNS 사이트 링크가 포함된 문장 삭제.
리트윗으로 중복되는 문장 삭제.
형태소 분석이 어려운 은어, 약어는 용이한 형태로 수정.

필터링 후 데이터
- 우울한 문장 : 1,297
- 우울하지 않은 문장 1,032

#### Environment
- Python : 3.6
- tf : 1.4.0
- keras : 2.1.5
- OS : Windows 10
- CPU : AMD Ryzen 7 2700
- RAM : 32GB

### Conclusion
GRU 모델이 약 92%로 가장 높은 정확도를 보였다.
성능 개선을 위해서는 더 많은 데이터를 확보하고 새로운 모델에 관한 연구가 필요할 것으로 보인다.