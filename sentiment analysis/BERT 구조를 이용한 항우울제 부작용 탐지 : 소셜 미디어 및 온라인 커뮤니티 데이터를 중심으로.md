## [BERT 구조를 이용한 항우울제 부작용 탐지 : 소셜 미디어 및 온라인 커뮤니티 데이터를 중심으로](https://library.yonsei.ac.kr/search/detail/CATTOT000002157730)

### Problem
항우울제 부작용 탐지

### Motivation
우울증 환자들은 증세와 치료에 관한 경험을 소셜 미디어나 온라인 커뮤니티에 공유하는 경우가 많다. 이에 온라인에서 항우울제에 대한 부작용 관련 내용을 탐지하고자 한다.

#### 약물 감시 연구에 활용할 수 있는 소셜 미디어 및 온라인 커뮤니티
- Ask a Patient : 의약품에 대한 평점과 리뷰를 남길 수 있는 약물 리뷰 사이트로 만족도를 매길 수 있고 부작용 경험에 대한 코멘트를 작성함.
- Twitter : 글로벌 소셜 미디어로 여론을 관찰 및 분석하는 분야에서 핵심 자원으로 활용.
- Reddit : 한국의 디시인사이드와 유사하게 세부 커뮤니티로 구성됨.
- WebMD : 의약품에 대한 평점과 리뷰를 남길 수 있는 최대의 온라인 건강 관련 커뮤니티.
- Drugs.com / Druglib.com : 의약품에 대한 평점과 리뷰를 남길 수 있는 약물리뷰 사이트

### Method
1. 데이터데서 부작용 문장을 분류한다.
2. 부작용 표현을 추출한다.
3. 부작용 표현을 정규화한다.

<img width="400" alt="스크린샷 2024-01-17 오후 3 12 23" src="https://github.com/K-Saaan/papers/assets/111870436/7cd8a435-9414-4c7c-8c61-39bdfbf7b9b2">

<img width="400" alt="스크린샷 2024-01-17 오후 3 12 57" src="https://github.com/K-Saaan/papers/assets/111870436/a2452921-25d9-40a3-af46-4c7650cb4b8e">

본 연구는 부작용 표현과 MedDRA의 표준화된 용어에 대한 정적인 임베딩을 생성하고 이에 대한 유사도를 계산하기 위해 SBERT(문장 단위의 임베딩을 얻기 위해 샴 네트워크 구조를 사용하여 문장 유사도 관련 데이터셋에서 파인튜닝한 모델)를 활용한다.

<img width="400" alt="스크린샷 2024-01-17 오후 3 16 10" src="https://github.com/K-Saaan/papers/assets/111870436/dec3366f-a5a5-4041-bdac-0290b024021d">

약물에 관한 사전학습이 잘 이루어진 BERT 모델이 없다고 판단하여 Intermediate 학습 단계를 추가했다. Intermediate는 사전학습과 파인튜닝 사이에 도메인이나 목표 작업과 가까운 데이터를 학습시키는 단계다. 

<img width="400" alt="스크린샷 2024-01-17 오후 3 20 19" src="https://github.com/K-Saaan/papers/assets/111870436/1e277b6b-c382-47d3-995c-260aa727bf1e">

Model
- BERT
- ClinicalBERT
- PubMedBERT
- SciBERT
- BIioBERT
- SapBERT

다양한 모델의 성능을 비교한다.
- BERT 모델을 바로 파인튜닝 한 경우
- 항우울제 약물 이름을 추가한 이후 MLM 학습을 진행한 모델을 파인튜닝 한 경우
- MLM Intermediate 학습을 한 경우

모든 모델은 Hugging space에서 제공하는 Transformers를 사용해 구현했다. 최적화 함수로는 AdamW를 사용했고 언급하지 않은 hyper-parameter는 기본값으로 두었다.

<img width="479" alt="스크린샷 2024-01-11 오후 4 01 38" src="https://github.com/kimdaehyuun/Quanters/assets/111870436/02a02085-ac8e-46b2-a05e-35dd3c5bf079">
<img width="474" alt="스크린샷 2024-01-11 오후 4 01 49" src="https://github.com/kimdaehyuun/Quanters/assets/111870436/10eade94-48ed-4f97-a6f1-7e7a1e9fd27e">

<img width="420" alt="스크린샷 2024-01-11 오후 4 02 26" src="https://github.com/kimdaehyuun/Quanters/assets/111870436/a7ed031e-f3a3-4a13-a2e0-b8825f31a5f1">

<img width="420" alt="스크린샷 2024-01-11 오후 4 03 44" src="https://github.com/kimdaehyuun/Quanters/assets/111870436/a09c5ce0-fbcc-43f9-b39f-96acf56f34ba">


### Experiment
#### Dataset
트위터와 레딧에서 25종의 항우울제 이름으로 검색 키워드를 설정하여 수집.

트위터에서 2006년 3월 21일 ~ 2022년 11월 1일 트윗 수집

<img width="519" alt="스크린샷 2024-01-11 오후 3 53 04" src="https://github.com/kimdaehyuun/Quanters/assets/111870436/0c1aef13-5024-4eb8-9ba8-a865b5577961">

레딧 : 7개의 subreddits (“antidepressants”, “depression”,
“depressionregimens”, “science”, “Nootropics”, “AskReddit”, “DrugNerds”) 에서 데이터를 수집

<img width="525" alt="스크린샷 2024-01-11 오후 3 54 33" src="https://github.com/kimdaehyuun/Quanters/assets/111870436/0578ca57-5469-4674-9e05-a5091a046e7b">

<img width="476" alt="스크린샷 2024-01-11 오후 3 56 00" src="https://github.com/kimdaehyuun/Quanters/assets/111870436/e2b010dd-1faa-4aeb-88a8-1a3a033e0883">

부작용을 언급하고 있는 문장은 1, 아닌 문장은 0으로 레이블링 했다. 중복된 데이터는 제거하고 모든 문자열을 소문자로 전환하고 정규식을 이용해 특수문자를 제거했다.

데이터 전처리 후 총 50,121개의 문장이 남았다.
- Train : 37,591
- val : 6,265
- test : 6,265
- 부작용 문장은 4,991개, 부작용이 아닌 문장은 47,130개로 약 1:9의 비율


### Conclusion
<img width="514" alt="스크린샷 2024-01-11 오후 4 04 20" src="https://github.com/kimdaehyuun/Quanters/assets/111870436/962f181c-b449-4d80-b5b4-0a939d1b593b">

SapBERT에 Intermediate 학습을 진행한 모델이 가장 좋은 성능을 보임.
부작용 표현 추출 모델 성능 실험에서는 SapBERT에 Intermediate 학습과 임베딩 조정을 진행한 모델이 좋은 성능을 보임.

분석 결과, 소셜 미디어에서 보이는 항우울제의 부작용은 기존 연구 내용 및 의료계의 치료 지침과 일치하거나, 항우울제 성분의 작용 기전에 의해 설명될 수 있었음.

본 연구는 소셜 미디어가 약물 부작용 탐지에 유용하게 활용될 수 있음을 보였다.
어휘 추가 및 intermediate 학습과 임베딩 조정이 모델의 성능을 향상하는 데 효과가 있음.

부작용 증상과 해당 약물 사이의 관계성을 확인하기는 어려움. 약물과 부작용에 대한 개체명 인식을 진행하고 이들 간의 관계를 예측해야함.
기저질환, 연령, 성별, 복용기간, 용량 등의 정보가 추가 된가면 더 깊이 있는 분석이 가능할 것이다.