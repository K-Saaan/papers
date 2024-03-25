## [Zero-Shot Learning Through Cross-Modal Transfer](https://arxiv.org/pdf/1301.3666.pdf)

### Abstract
본 연구에서는 객체에 대한 training data가 없어도 이미지를 식별할 수 있는 모델을 소개한다. unseen categories를 식별하기 위해 unsupervised large text corpora가 필요하다. 이전의 연구에서는 unseen class만 식별할 수 있었다. 본 연구는 semantic space에서 이상치를 감지하여 두개의 seperate recognition model을 사용해 SOTA를 달성했다. 또한, 단어나 이미지에 대해 정의할 필요가 없다.


### 1. Introduction
Zero-Shot learning은 본 적 없는 클래스를 분류할 수 있도록 하는 것이다. 이는 다양한 상황에서 사용된다. 자동차나 전자기기처럼 label이 없고 신상품이 자주 출시되는 상품에서 많이 사용된다. AI는 카테고리별로 학습된 정보가 있어야 식별할 수 있지만 인간은 몇 가지 정보만으로 세단과 SUV를 식별할 수 있다. Zero-Shot Learning은 이러한 인간의 능력을 모델링하는 것이다.

Zero-Shot learning은 seen class, unseen class 모두 예측할 수 있다. 고양이 사진이 있으면 모델은 사진이 고양이인지 아니면 개나 말처럼 학습된 데이터셋인지 알아낸다. 본 모델은 다음의 두가지 main idea에 기초한다.

1. 이미지를 신경망 모델로 학습된 단어의 의미적 공간에 매핑한다.
2. 분류기는 test 이미지를 seen class로 분류하는 것을 선호하기 때문에 모델은 새로운 이미지가 seen class에 해당하는지 아닌지를 결정하는 이상치 탐지 확률을 통합한다. 이상치 탐지를 사용하여 이미지가 seen class인지 unseen class인지 분류한다.  

<img width="815" alt="스크린샷 2024-03-25 오전 10 20 08" src="https://github.com/K-Saaan/papers/assets/111870436/4d9f6531-586b-4205-a774-cefa28170b4c">

[Figure 1] 이미지가 semantic space에 매핑되어 있는 모습 (같은 class끼리 군집을 이루고 있는 것을 확인할 수 있다.)
- seen class : dog, hores, auto 
- unseen class : cat, truck

cat 이미지가 입력으로 들어왔을 때 unseen classdls cat에 속할지 seen calss에 속할지 정한다.



### 2. Related Work
관련된 5개 연구와의 차이점과 연관성을 알아보자.

#### Zero-Shot Learning
training data에는 없는 unseen class를 의미적으로 매핑해 예측한다는 점에서 본 연구와 가장 비슷하다.
사람들이 단어를 떠올릴 때의 fMRI scan을 feature space에 매핑하고 이를 사용해 분류한다. 학습 데이터에 없던 unseen class라도 매핑되는 word에 대해 semantic feature가 예측 가능하고, unseen class 간의 구분도 가능하다. 하지만 새로운 test instance가 입력되면 seen인지 unseen인지 분류하지 못한다. 이러한 부분을 개선하기 위해 본 연구에서는 outlier decetion을 사용했다.

#### One-Shot Learning
이미지 인식 분야에서 training example의 수가 적을 때 사용하는 방법이다. 참고 논문에서는 Bayesian layer model과 Hierarchical deep model을 이용해 이미지 분류를 학습한다. feature representation과 모델 파라미터를 공유해서 사용한다. low-level image feature(픽셀의 강도나 색상처럼 컴퓨터가 이미지를 식별하고 분류하는 데 사용하는 특징)를 학습 후 knowledge transfer를 한다. 자연어로부터 cross-modal knowledge transfer를 하기 때문에 학습데이터가 필요하지 않다.

#### Knowledge and Visual Attribute Transfer
잘 설계된 sementic attribute를 이용해 이미지를 분류한다. 본 연구에서도 attribute를 사용하지만 이미지의 의미적인 부분에 대해서는 corpus에서 학습된 단어의 분포적 특성만 가지고 있고 train image가 0인 카테고리와 1,000개인 카테고리의 분류가 가능하다는 점이 다르다.

#### Domain Adaptation
한 도메인에서 학습된 분류기를 다른 도메인에 적용시키는 방법이다. 이러한 방법은 한 도메인에 training data가 많고 다른 도메인에는 적거나 없을 때 유용하다. 예를 들어 영화 리뷰 분류기를 책 리뷰에 적용한다면 관련되어 있지만 각 클래스에 대한 데이터가 있기 때문에 작업 라인이 달라 feature가 다를 것이다. 본 연구에서도 domain adaptation이 필요하다.

#### Multimodal Embeddings
Multimodal embedding은 음성, 비디오, 이미지, 텍스트와 같이 여러 소스에서 나온 정보들을 연관짓는 것이다. 단어와 이미지를 같은 공간에 투영해 annotation과 segmentation 부분에서 SOTA를 달성했다. 본 연구에서 semantic word representation 학습을 위해 unsupervised large text corpora를 사용했다.

<img width="600" alt="스크린샷 2024-03-25 오전 11 49 25" src="https://github.com/K-Saaan/papers/assets/111870436/e80da496-943b-4d25-a09a-caa710c4a751">



### 3. Word and Image Representations
distributional approach는 단어들 간 의미적 유사성을 확인할 때 자주 사용되는 방법이다. 단어가 분포적 특성의 벡터로 표현되고 대부분 co-occurrences 방식을 사용한다. 단어 벡터는 50차원의 pre-trained 벡터로 초기화 되고, 이 모델은 위키피디아 text를 사용해 각 단어가 context에서 발생할 가능성을 예측하여 학습한 것이다.


### 4. Projecting Images into Semantic Word Spaces
이미지의 semantic relationship, class membership 학습을 위해 image feature 벡터를 50차원의 단어 공간에 투영했다. 

<img width="348" alt="스크린샷 2024-03-25 오후 12 18 10" src="https://github.com/K-Saaan/papers/assets/111870436/ed035c9b-44ea-4938-bf9c-2fc0a05f7ff6">

[식 (1)]

이미지와 단어 간 매핑 훈련을 위해 위 식을 최소화해야 한다.

$$Y_s : seen\ class$$

$$ Y_u : unseen\ class $$

$$ W_y : class\ name에\ 대응되는\ word\ vector $$

$$ x^(i) : training\ image $$

$$ \theta : neural\ network\ 가중치 $$

<img width="739" alt="스크린샷 2024-03-25 오후 12 28 04" src="https://github.com/K-Saaan/papers/assets/111870436/2d88139d-bb6a-4004-96cc-86d76e4e83a2">

[Figure 2] semantic word space를 T-SNE을 사용해 시각화 했다.
각 class별로 대응되는 word vector끼리 군집을 이루고 있다. unseen class인 cat, truck는 군집과 떨어져 있지만 truck은 automobile과 cat은 dog와 가까이 위치해 있다. 이는 목적함수와 학습방식으로 의미적으로 유사한 class와 가깝게 위치하고 있다. 이런 방식으로 unseen class의 outlier detection할 수 있고 이를 zero-shot word vectorㅎ라고 분류할 수 있다.

### 5. Zero-Shot Learning Model
본 연구의 모델은 test set의 이미지 x에 대한 조건부 확률 p(y|x)를 예측하는 것이다. 일반적인 분류기는 training example에 없던 class를 예측할 수 없기 때문에 binary visibility random variable(V)을 사용해 어떤 이미지가 seen, unseen인지 예측한다. 

<img width="583" alt="스크린샷 2024-03-25 오후 1 53 02" src="https://github.com/K-Saaan/papers/assets/111870436/8663337e-269b-4b4a-b028-98f4fb1ac2fa">

[식 (2)]

$$ y \in y_s \cup y_u : seen and unseen \ 클래스에 \ 대한 \ y값 $$

$$ x \in X_t : test \ set \ 이미지 $$

$$ f \in F_t : test \ set semantic \ vectors $$

$$ X_s : seen \ 클래스에 \ 대한 \ 모든 \ training \ set \ 이미지의 \ feature \ vectors $$

$$ F_s : 각 \ X_s에 \ 대응하는 \ semantic \ vectors $$

$$ F_y : y의 \ 클래스에 \ 대한 \ semantic \ vectors $$

$$ V \in {s, u} : seen \ and \ unseen \ 클래스에 \ 대한 \ visibility \ 변수 $$

$$ \theta : image \ feature \ vector를 \ d차원 \ semantic \ word \ space에 \ 매핑하기 \ 위한 \ neural \ network \ 파라미터 $$

- 이상치 탐지 점수를 임계값으로 지정하여 사용

<img width="452" alt="스크린샷 2024-03-25 오후 2 03 07" src="https://github.com/K-Saaan/papers/assets/111870436/5ba55aa7-255d-4407-bf97-8b082efa98bc">

[식 (3)]
특정 Threshold T를 설정해 새로운 이미지 x의 이상치가 T 이하면 unseen으로 판단한다.

- seen과 unseen에 대한 weighted combination of  classifier를 사용해 클래스에 대한 조건부 확률을 구함.
Fig2를 보면 unseen데이터 중에서 이상치가 아닌 것으로 측정되는 경우가 많은 것을 볼 수 있다. 본 방법은 outlier detection에 보수적이며, seen 클래스에 대해 높은 정확도를 보인다. 따라서 각 test클래스에 대한 이상치 확률을 얻은 다음 seen, unseen 클래스 모두에 대한 분류기의 가중치 조합을 사용한다.

seen class의 경우 softmax classifier를 사용했다.
unseen class의 경우 각 novel class word vectors에 isometric Gaussian을 가정해 likelihood에 따라 클래스를 분류한다.
isometric Gaussian : unseen 클래스의 word vector와 맵핑된 이미지 벡터 사이에서 거리가 가장 가까운 클래스로 분류해주는 모델

### 6. Experiments
본 연구 실험에서는 CIFAT10 데이터셋을 사용했다. CIFAT10은 10개의 클래스로 구성되어 있고 각 클래스는 5,000개의 32 x 32 x 3 RGB 데이터로 구성되어 있다. 데이터를 seen, unseen으로 구분하기 위해 CIFAT10 중 2개의 클래스를 unseen으로 가정해서 사용했다.(2개의 클래스를 학습에 사용하지 않고 test에서 사용)

#### 6.1 Zero-shot Classes Only
본 section에서 두 zero-shot class 간의 분류를 비교한다. zero-shot 클래스와 유사한 클래스가 없다면 성능이 무작위에 가깝다는 것을 확인할 수 있다. 예를 들어, 고양이와 개를 훈련에서 제외하면 test에서 고양이가 있을 때 다른 클래스에 유사한 클래스가 없기 때문에 zero-shot 분류의 성능이 좋지 않다. 개를 training에서 사용하면 고양이가 입력됐을 때 좋은 성능으로 매핑될 수 있다.

<img width="759" alt="스크린샷 2024-03-25 오후 3 13 00" src="https://github.com/K-Saaan/papers/assets/111870436/c6143228-b99b-4b95-8dc9-fc2abdfcef5b">

[Figure 3] 이상치 탐지를 위한 다양한 컷오프에서의 성능.
다양한 Threshold에 대한 seen class와 zero-shot class 쌍의 classification accuracy

#### 6.2 Zero-shot and Seen Classes
그림 3에서 test시 이미지를 seen 또는 unseen 클래스로 분할하는 임계값에 따라 약 80%의 정확도를 얻을 수 있음을 볼 수 있다.
70%의 정확도, unseen 클래스에서 30%에서 15% 사이의 정확도로 분류될 수 있고 무작위 확률은 10%이다.

### 7. Conclusion
본 연구는 딥러닝 기반으로 zero-shot 분류를 위한 새로운 모델을 도입했다. 두 가지 핵심 아이디어는
1. semantic word vector가 나타내는 표현은 이미지와 텍스트 간의 transfer knowledge를 가능하게 한다.
2. 베이지안 프레임워크는 zero-shot과 일반 분류를 하나의 프레임워크로 묶을 수 있다.













