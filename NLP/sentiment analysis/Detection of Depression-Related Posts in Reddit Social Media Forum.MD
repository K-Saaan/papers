## [Detection of Depression-Related Posts in Reddit Social Media Forum](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8681445)

### Problem
많은 국가에서 우울증은 여전히 제대로 진단되지 않고 있다. 심한 경우 자살로 이어질 수 있으며, 사회적 시선에 의해 치료를 거부하기도 한다.
온라인이 활성화되면서 온라인에서 우울증을 감지하기 위한 연구가 많이 진행됐다.
BOW, N-grams, LIWC, LDA와 같은 단일 세트 기능을 사용하여 식별하기도 했다. 다른 연구에서는 개별 기능의 성능을 다양한 학습 분류기와 비교한다. 최근 연구에서는 N-grams+LIWC, BOW+LDA, TF-IDF+LDA와 같이 단일 기능을 조합하여 정확도를 올린다.

본 논문에서는 적절한 기능과 조합을 통해 성능 향상에 대한 해결책을 찾는 것을 목표호 한다.

### Motivation
다양한 모델과 조합으로 우울증을 식별하는 연구가 진행되었지만 여전히 개선할 부분이 남아있다.

### Method
<img width="471" alt="스크린샷 2024-01-18 오후 2 18 59" src="https://github.com/K-Saaan/papers/assets/111870436/19bb2c7d-c6e9-4700-b448-885fe11a3870">

1. 게시물의 내용을 특성화하기 위해 우울증 식별에 적용되는 언어적 특징을 선택한다.
2. 텍스트에서 추출한 상관관계 유의성, 숨은 주제, 단어 빈도를 분석한다.
    - 상관관계 : LIWC 사전과 언어적, 심리적, 개인적 관심사에 중점을 둔다.
    - 주제 : LDA
        - 70개의 주제로 제한될 때 성능이 가장 좋음.
        - 주제 선택 시 최소 10개 이상의 게시물에 등장하는 단어만 고려함.
    - 단어 빈도 : TF-IDF를 기반으로 N-gram(유니그램과 바이그램)을 사용.
3. 5가지 텍스트 분류기법을 설정하고, 추출된 데이터를 이용하여 우울증 식별을 수행한 후 성능을 비교한다.

<img width="350" alt="스크린샷 2024-01-18 오후 2 26 57" src="https://github.com/K-Saaan/papers/assets/111870436/19ff9b47-38cc-4c73-b2b4-4cb022cfc548">

<img width="348" alt="스크린샷 2024-01-18 오후 3 00 42" src="https://github.com/K-Saaan/papers/assets/111870436/3e7035a0-1061-4c57-a99d-9683333f8150">
<img width="360" alt="스크린샷 2024-01-18 오후 3 00 50" src="https://github.com/K-Saaan/papers/assets/111870436/25597c87-23f6-43d2-b472-aa79707e1858">

#### Model
- Logistic Regression
- Support Vector Machine
- Random Forest
- Adaptice Boosting
- Multilayer Perceptron (4개, 16개 레이어 적용)

<img width="486" alt="스크린샷 2024-01-18 오후 3 09 18" src="https://github.com/K-Saaan/papers/assets/111870436/5f473d23-649f-48ca-aad6-09db59a2c4e1">


### Experiment
#### Dataset
Reddit에서 수집
데이터 셋은 Inna Pirina et al.에 의해 구축
우울증 게시물 : 1,293
표준 게시물 : 548

1. 게시물을 개별 토큰으로 나눈다.
2. URL과 같은 불필요한 부분을 삭제한다.
3. 단어를 어근 형태로 줄이고 유사한 단어를 그룹화하기 위해 형태소 분석을 적용한다.

<img width="244" alt="스크린샷 2024-01-18 오후 2 26 31" src="https://github.com/K-Saaan/papers/assets/111870436/86322fa5-32d4-47ba-a777-c8f84a472d5c">


### Conclusion
우울증이 있는 텍스트의 경우 현재와 미래에 더 중점을 두고 자신에 대한 집착, 슬픔, 불안, 분노, 적대감, 자살 충동과 관련된 단어가 있었다.
MLP 분류기가 accuracy 91 , F1 0.93으로 가장 높은 성능을 보였다.
단일 기능 세트 중 가장 좋은 것은 바이그램이었다.