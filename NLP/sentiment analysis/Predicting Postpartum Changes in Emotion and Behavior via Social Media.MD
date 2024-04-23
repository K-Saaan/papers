## [Predicting Postpartum Changes in Emotion and Behavior via Social Media](https://erichorvitz.com/predicting_postpartum_changes_chi_2013.pdf)

### Problem
출산 전 산모를 관찰하여 산후 행동과 기분의 변화를 예측한다.

### Motivation
잘 알려지지 않은 산후 우울증에 대해 연구한다. 산후 우울증이 있는 산모를 식별하여 사전에 문제를 인식하고 해결할 수 있도록 연구하게 되었다.

### Method
산모의 행동과 기분을 특성화하기 위해 네 가지 척도를 사용한다.
##### Engagement
    - 산전 및 산후 기간 동안 산모가 올린 하루 평균 게시물 수
    - 답장 게시물
    - 리트윗 비율
    - 공유한 링크
    - 특정 날짜에 보낸 게시물의 비율
##### Ego-network
    - 팔로우 수
    - 팔로어 수

##### Emotion
    - 긍정 및 부정적 감정의 활성화 및 지배력을 고려

##### Linguistic Style
    - 게시물의 언어적 스타일 고려(LIWC 참조)

과적합을 방지하고 기능 중복 및 상호 작용 제거를 위해 PCA와 정규화된 Random Forest를 사용한다.
다양한 분류기를 비교하여 최적의 분류기를 찾고자 한다.
- LDA
- QDA
- Naive Bayes
- K-NN
- Decision trees
- SVM
위 분류기 중 SVM의 성능이 가장 좋았다.

<img width="336" alt="스크린샷 2024-01-18 오후 4 18 02" src="https://github.com/K-Saaan/papers/assets/111870436/611e81be-62fe-4800-8a05-30e17b348b25">



### Experiment
#### Dataset
데이터 수집 플랫폼 : 트위터
- 트위터에서 최근 출산한 아이에 대한 언급이 있는 게시물을 찾는다. (2009년 ~ 2012년)
<img width="336" alt="스크린샷 2024-01-18 오후 3 49 21" src="https://github.com/K-Saaan/papers/assets/111870436/60e8da30-4d95-43d0-8aad-fbd8ba0b638e">

- 인구 조사 데이터를 기반으로 이름을 통해 성별 추론을 학습한다.
- 앞서 수집한 트위터에서 엄마를 찾는다.

2011년 3월 ~ 2012년 7월 사이 376명의 산모가 올린 36,948개의 게시물과 보내온 40,426개를 데이터셋으로 사용.

### Conclusion
산전 행동에 대한 관찰을 통해 출산 후 변화에 대해 71%의 정확도를 확인할 수 있었다.
산후 2~3주 데이터를 추가로 화룡ㅇ하면 80~83%의 정확도를 얻을 수 있었다.