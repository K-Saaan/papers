## [A  Deep  Learning-based  Depression  Trend  Analysis  of Korean  on  Social  Media](https://accesson.kr/kosim/v.39/1/91/11032)

### Problem
한국어 텍스트에서 우울증을 식별하고자 한다.

### Motivation
- 텍스트에서 감성분석을 통해 우울증을 식별하는 기존의 연구들은 영어 텍스트를 사용해서 수행되었다.
- 한국어 데이터셋을 사용해 한국어 텍스트에서 우울증을 식별하고자 함.

### Method
- LSTM
- CNN
- RNN
- KorBERT

<img width="600" alt="스크린샷 2024-01-11 오후 4 29 09" src="https://github.com/kimdaehyuun/Quanters/assets/111870436/e5c77e50-eed2-40f2-bea9-175c33950488">

<img width="750" alt="스크린샷 2024-01-11 오후 4 28 11" src="https://github.com/kimdaehyuun/Quanters/assets/111870436/2fa489b7-650f-4437-9099-f2ec0c43df68">


### Experiment
#### Dataset
1. 네이버 지식인, 네이버 블로그, 하이닥, 트위터에서 우울, 우울증과 같은 검색어로 2017년 1월 ~ 2019년 12월까지의 데이터 크롤링하여 수집.
<img width="460" alt="스크린샷 2024-01-11 오후 4 21 49" src="https://github.com/kimdaehyuun/Quanters/assets/111870436/3bf03b7d-12e9-480d-8579-5aa5036d85c9">

2. 수집한 데이터 442,859건 에서 임의표본추출을 통해 4,604건을 추출하여 DSM-5(우울 장애 진단 기준)를 사용해 주석을 달았다. (주석 작업을 마친 데이터셋을 정신과 의사에게 검증을 받았다.)

3. 멀뭉치 클래스별 특성을 확인하기 위해 TF-IDF 분석과 동시출현어 분석을 실시.

4. 우울 분류 모델을 생성하기 위해 Word2Vec을 사용한 단어 임베딩, 사전 기반 감성 분석, LDA 토픽 모델링 수행

5. 임베딩된 텍스트와 감성 점수, 토픽을 산출하여 텍스트 특징으로 사용.


### Conclusion
- 내장된 텍스트만을 사용한 경우 정확도가 낮았다.
- 다른 텍스트와 결합한 경우 성능이 향상됐다.
- 다양한 텍스트 특징을 사용할수록 성능이 향상됐다.

