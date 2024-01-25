## [Supervised machine learning models for depression sentiment analysis](https://pubmed.ncbi.nlm.nih.gov/37538396/)

### Problem
머신러닝 모델과 감성분석 기술을 사용해 사용자의 게시물에서 초기에 우울증 수준을 예측하고자 한다.

### Motivation
전 세계적으로 정신 질환 문제는 심각하지만, 특히 우울증의 비율은 계속해서 높아지고 있다.

### Method
#### Model
- SVM
- LR
- XGB Classifier
- RF

### Experiment
#### Dataset
Kaggle에서 4개의 별도 Twitter 데이터 세트를 수집하여 하나로 병합.
NLTK 사용하여 다음 작업 수행
- 토큰화
- on, at, the와 같은 전치사 및 관사를 제거
- 형태소 분석 : 게시글에 있는 단어의 어근 식별
- Lemmnatization(텍스트 정규화) : 형태소 분석과 비슷하지만 어근에 의미가 있음
- 바이그램/트라이그램 생성

#### 분류
1. 트윗을 1~4의 범위로 분류하는 수치 분류
2. 극성에 따라 부정, 긍정, 중립으로 분류
긍정, 부정, 중립으로 3중 분류되어 있는 기존의 트윗을 긍정 = 0 / 부정 = 1 / 중립 = 2로 구분한다.
이후 1~4단계로 분류하는 수치 분류를 수행했다.

데이터 정리, 토큰화, 불용어 제거, 형태소 분석, 표제어 분석, 바이그램 생성, 감정 분류, 중복 제거

<img width="826" alt="스크린샷 2024-01-25 오후 4 45 52" src="https://github.com/K-Saaan/papers/assets/111870436/77651215-f9cb-4500-ae4f-4ce4214c5bfe">

<img width="796" alt="스크린샷 2024-01-25 오후 4 46 32" src="https://github.com/K-Saaan/papers/assets/111870436/ed41a214-1700-49a5-8070-0fd8af711d68">


### Conclusion

<img width="670" alt="스크린샷 2024-01-25 오후 4 50 16" src="https://github.com/K-Saaan/papers/assets/111870436/6e61db7e-b917-4fa0-9649-a47d0c02e55c">

<img width="851" alt="스크린샷 2024-01-25 오후 4 50 47" src="https://github.com/K-Saaan/papers/assets/111870436/dfbec21a-85d2-48c9-a4b1-b90a6c2f82e7">
47d0c02e55c">
