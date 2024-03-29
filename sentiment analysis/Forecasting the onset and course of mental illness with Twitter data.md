## [Forecasting the onset and course of mental illness with Twitter data](https://www.nature.com/articles/s41598-017-12961-9)

### Problem
SNS를 통해 우울증, PTSD, 자살 생각과 같은 상태를 진단하는 연구가 있지만 아직 초기 단계에 있어 개선이 필요하다. 본 연구에서는 트위터에서 우울증과 PTSD를 예측하고 추적하기 위한 개선된 방법을 제시한다.

### Motivation
PTSD는 흔하지 않지만 우울증과 동반되가 경우가 많다. 우울증과 PTSD의 조기 선별 및 진단을 통해 비용과 노동을 줄이는 것과 같은 긍정적인 영향을 미칠 수 있을 것이다.
첫번째 우울증 진단 날짜 이전에 작성된 트윗을 통해 우울증 게시물에 대한 학습을 한다.
### Method
#### ML
Random Forests가 최고의 성능을 보임
하이퍼파라미터 최적화를 위해 5번의 교차 검증 사용.
- 우울증 모델 : 10개 중 1건의 False 발생
- PTSD 모델 : 10개 중 1건의 False 발생
<img width="600" alt="Figure 1" src="https://github.com/kimdaehyuun/Quanters/assets/111870436/37ceaf99-aaa3-4332-b72a-251ae139fb4d">

#### 시계열 분석
소셜 미디어 상의 언어 사용 패턴으로부터 사람들의 상태를 추론하기 위해 Hidden Markov 모델(HMM은 관찰 가능한 데이터를 사용하여 시간에 따른 잠재적 상태를 추정한다. 우울증을 가진 사람이 얼마마 많은 단어를 사용하는지에 대한 관심보다 이로 인해 우울증에 대한 접근을 가능하게 할 것이다라는 기대를 가짐)을 사용했다. 
우울증 환자와 그렇지 않은 사람의 HMM 데이터를 비교했다. HMM의 결과와 실제 데이터와 비슷하다면, 대상자의 상태가 HMM에 의해 잘 표현되었다고 볼 수 있다. 


### Experiment
#### Dataset
설문조사 플랫폼 Qualtrics를 사용하여 설문을 했다.
대상자는 우울증이나 PTSD의 영향을 받은 사람과 그렇지 않은 사람을 대상으로 했다.
우울증 수준을 확인하기 위해 CES-D(역학 연구 우울증 척도 센터) 설문지를 사용했다.
대상자의 트위터 정보를 수집했다. (각 대상자별 최대 3,200 트윗까지 수집)
우울증 : 총 204명의 사용자로부터 279,951개 수집
PTSD : 총 174명의 사용자로부터 243,775개 수집

- 필터링
게시물이 총 5개 미만인 참가자는 제외
CES-D(우울증) 점수가 21점 이하 제외
TSQ(PTSD) 점수가 5점 이하 제외

트윗 메타데이터 분석을 통해 트윗당 평균 당어 수, 리트윗인지 여부, 다른 사람의 트윗에 대한 응답인지 여부를 평가.
labMT, LIWC 2007, ANEW 유니그램 감정도구를 사용해 트윗 언어의 행복도를 정량화하는 데 사용.

### Conclusion
우울증을 앎는 사람은 진단 9개월 전의 기간동안 우울증에 걸릴 확률이 건강한 사람에 비해 약간 더 높았다.
진단 3개월 전 우울증 환자는 우울증 상대에 있을 확률이 크게 증가했다.
우울증 진단 후 3~4개월 이후 우울증 확률이 감소하기 시작했다. 이는 치료 프로그램에서의 평균 개선 기관과 유사하다.
건강한 사람은 꾸준히 낮은 우울증 확률을 보였고 이는 18개월 동안 큰 변화가 없었다.

PTSD 환자의 경우 유발한 사건으로부터 몇 달 이내에 정상 범주를 벗어났다.
진단까지의 평균 시간은 586일 정도였다.
PTSD의 확률이 감소하는 것은 진단 이후 낮아졌다.

우울증 환자와 건강한 사람의 가장 큰 차이는 부정적인 단어 사용 증가였다.
'절대로', '감옥', '살인', '죽음' 등
긍정적인 언어 감소도 보였다. '행복한', '해변', '사진'의 출현이 감소했다.