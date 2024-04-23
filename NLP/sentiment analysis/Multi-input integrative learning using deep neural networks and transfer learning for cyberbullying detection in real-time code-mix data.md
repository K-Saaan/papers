## [Multi-input integrative learning using deep neural networks and transfer learning for cyberbullying detection in real-time code-mix data](https://link.springer.com/article/10.1007/s00530-020-00672-7)

### Problem
사이버 괴롭힘의 확대와 심각함을 방지하기 위해 사전에 감지하고자 함.

### Motivation
온라인 환경이 발전하고 많은 사람들이 사용하면서 사이버 괴롭힘 사례가 계속해서 증가하고 있다. 괴롭힘의 범위가 사이버까지 확대됐기 때문에 쉬운 문제로 볼 수는 없다. 인스타크램(42%), 페이스북(37%), 스냅챗(31%) 순으로 발생했다. 

### Method

#### 특징 추출
MIIL-DNN이라는 영어, 힌디어, 활자체 입력을 사용해 훈현된 모델 생성.
영어 : CapsNet
힌디어 : bi-LSTM
이외 : MLP

GloVe와 fastText를 사용하여 특징 추출을 한다.

<img width="561" alt="스크린샷 2024-01-18 오후 5 20 12" src="https://github.com/K-Saaan/papers/assets/111870436/cd5346f5-2a6a-4339-9c56-386097e6f5ee">


<img width="554" alt="스크린샷 2024-01-18 오후 5 19 38" src="https://github.com/K-Saaan/papers/assets/111870436/cbe65840-a1f6-4e34-be18-bbfac92f1f80">


10번의 교차 검증을 수행하고 AUC를 계산.
Scikit-learn, keras 라이브러리 사용.

<img width="731" alt="스크린샷 2024-01-18 오후 5 15 21" src="https://github.com/K-Saaan/papers/assets/111870436/603f1e32-8633-4bd8-81dd-10f4d8939647">


### Experiment
#### Dataset
정치, 공인, 연예 등 영역에서 특정 해시태그와 키워드를 선정(힌디어 + 영어가 혼합된 언어 사용)하여 Twitter와 Facebook에서 데이터를 수집했다.
Facebook : 프로필 기반 게시물
- 댓글 추출 : GraphAPI 사용
Twitter : #Ind VS Pak, # Beaf Ban, #movies 등 가장 유행하는 주제에 속하는 주제 기반 트윗을 수집.
- Tweepy 사용

<img width="733" alt="스크린샷 2024-01-18 오후 5 02 50" src="https://github.com/K-Saaan/papers/assets/111870436/d44aa895-ddec-477f-a69a-9be3493bd1b6">

평균 게시물 길이

<img width="733" alt="스크린샷 2024-01-18 오후 5 03 38" src="https://github.com/K-Saaan/papers/assets/111870436/5012a1e1-9cb2-4b6d-93db-471ddb6adaf2">

평균 단어 길이
<img width="729" alt="스크린샷 2024-01-18 오후 5 04 27" src="https://github.com/K-Saaan/papers/assets/111870436/347d400d-1db2-4c22-973a-2d204ff0bbbc">

#### 데이터 전처리
- 태그, 숫자, URL, 언급, 불용어 및 구두점 제거
- 맞춤법 검사, 표제어 분석 및 형태소 분석
- 소문자로 변환
- SMS 사전을 사용하여 속어 및 이모티콘 대체
NLTK(Python Natural Language Toolkit)의 TreebankWordTokenizer를 사용해 토큰화 수행.

Facebook : 괴롭힘 3,350 괴롭힘 아님 3,225
Twitter : 괴롭힘 3,350  괴롭힘 아님 3,150


### Conclusion
<img width="568" alt="스크린샷 2024-01-18 오후 5 16 36" src="https://github.com/K-Saaan/papers/assets/111870436/55d13d43-3158-458d-8cc9-787910e56935">

기존 모델과의 비교

<img width="732" alt="스크린샷 2024-01-18 오후 5 17 15" src="https://github.com/K-Saaan/papers/assets/111870436/aad23596-74cb-4185-b42e-d6cd3b46f355">

1. 교차 언어 데이터 셋에도 적용할 수 있다.
2. 제안된 통합 학습 네트워크 MIIL-DNN은 다국어 융합을 사용해 결과를 출력한다.