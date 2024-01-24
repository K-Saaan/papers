## [A Novel Co-Training-Based Approach for the Classification of Mental Illnesses Using Social Media Posts](https://ieeexplore.ieee.org/document/8901145)

### Problem
정신건강에 대해서는 소셜 플랫폼을 통해 이야기하는 것이 더 개방적이다. 온라에 있는 데이터를 활용해 정신 징환 지표를 추출하고자 한다.
게시물에서 우울증, 불안, 양극성 장애 및 ADHD와 같은 정신 장애를 분류하도록 한다.

### Motivation
세계의 많은 사람들이 정신 질환 문제를 가지고 있다. 의학이 발전하고 있지만 정신 질환의 경우 공개하길 꺼리고 있다. 온라인에서 이야기 되는 사례가 많기 때문에 온라인에서 정신 질환을 추출하고자 한다.

### Method

#### 특징 추출
TF-IDF 단어 분해를 사용해 특징 추출

#### 특징 선택
가지치기(pruning) / 클러스터링(clustering)
데이터 세트에 가지치기 및 카이제곱 적용(카이제곱이 더 뛰어나다고 함. 특성과 목표사이의 카이제곱을 계산)

<img width="358" alt="스크린샷 2024-01-24 오후 5 33 28" src="https://github.com/K-Saaan/papers/assets/111870436/241d5334-0fa5-496a-9a6e-cd2a0cf226a8">

#### train test split
train : 80
test : 20

#### Model
게시물과 댓글이 모두 포함된 데이터 세트 생성 (단 댓글은 라벨링 하지 않음)
- SVM
- RF
- NB
- SVM, RF, NB 앙상블

<img width="685" alt="스크린샷 2024-01-24 오후 5 39 07" src="https://github.com/K-Saaan/papers/assets/111870436/d8495084-ec78-4ed0-beb5-0119b5eb6174">


### Experiment
#### Dataset
수집 플랫폼 : Reddit
하위 레딧 : Depression, Anxiety, ADHD, Bipole
수집 게시물 수 : 3,922개 (게시물 당 상위 5개의 댓글도 수집)

##### 전처리
1. 데이터 세트에서 빈 행 또는 null 데이터 제거
2. 모든 텍스트 데이터를 소문자로 변경 (동일한 단어가 서로 다르게 간주될 수 있기 때문)
3. 토큰화
4. 불용어 제거
5. 형태소 분석 또는 Lemmatization (각 단어의 어근을 생성하고 동일한 개념적 의미를 가진 단어를 그룹화)
6. 알파벳이 아닌 문자 제거

### Conclusion
#### NB

<img width="367" alt="스크린샷 2024-01-24 오후 6 02 04" src="https://github.com/K-Saaan/papers/assets/111870436/c9072d3d-5cd6-44db-869a-1ce3b1806cf5">

#### RF

<img width="371" alt="스크린샷 2024-01-24 오후 6 02 52" src="https://github.com/K-Saaan/papers/assets/111870436/6216e0d7-d3c3-4c2a-89b2-7d112af4cd58">


#### SVM

<img width="368" alt="스크린샷 2024-01-24 오후 6 03 17" src="https://github.com/K-Saaan/papers/assets/111870436/7214b6a2-5f64-402f-92bf-9d2ec8911d8a">

#### Co-Training

<img width="734" alt="스크린샷 2024-01-24 오후 6 04 40" src="https://github.com/K-Saaan/papers/assets/111870436/245f322f-31fa-4603-9efa-1765e05a5bd5">

개별 사용에 비해 Co-Training의 성능이 더 좋다.

