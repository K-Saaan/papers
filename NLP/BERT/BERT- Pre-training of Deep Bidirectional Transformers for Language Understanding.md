## [BERT- Pre-training of Deep Bidirectional Transformers for Language Understanding.md](https://arxiv.org/abs/1810.04805?source=post_page)

### Abstract
새로운 언어 모델 BERT(Bidirectional Encoder Representations from Transformers)에 대한 소개. 
최근 언어모델과 다르게 BERT는 unlabeled text에서 deep bidirectional representations pre-train이 가능하다.
BERT는 unlabeled text에서 pre-train을 진행 후 이를 특정 downstream task(with labeled data)에 fine-tuning(transfer learning)을 한 모델이다.
다른 언어모델인 Bidirectional LSTM이나 ELMo 모델에서도 bidirectional이 나오지만 BERT에서는  deep bidirectional을 통해 차별성을 강조했다.
BERT는 하나의 output layer를 pre-trained BERT 모델에 추가하면 질의응답, 언어추론과 같은 주요 task에서 SOTA를 달성할 수 있다.(BERT 모델의 확장성이 넓고 기존의 모델보다 뛰어난 성능을 보여줌)
- SOTA(State of the art) 현재 최고 수준의 결과



### 1. Introduction
Language model pre-training은 좋은 성능을 보인 연구가 많다.
대표적으로 natural language inference(NLI 자연어 추론), paraphrasing과 같은 문장 레벨 task와 QA, 개체명 인식과 같은 token 레벨 task에서 두각을 보였다.
문장 레벨 task는 두 문장의 관계를 분석하여 예측하는 것을 목표로, 토큰 레벨 task는 token 단위의 fine-grained output(output을 내기 위해 작은 단위의 output 프로세스로 나눈 뒤 수행)을 만들어야 한다는 특징이 있다.

down stream task에 pre-trained language representation을 적용하는 방법은 크게 feature based approach, fine-tuning approach 두가지가 있다.

#### feature based approach
- 대표적으로 ELMo가 있다. ELMo는 pre-trained representation을 하나의 추가적인 feature로 활용한 task-specific architecture를 사용한다.

<img width="223" alt="스크린샷 2024-03-08 오후 12 17 04" src="https://github.com/K-Saaan/papers/assets/111870436/624ba0c4-7526-4cca-9573-59d9c487a662">

[ELMo 구조]

bidirectional language model을 통해 얻은 representation을 embedding vector와 concat 해준다.


#### fine-tuning approach
- OpenAI의 GPT가 fine-tuning approach을 사용했다. GPT는 task-specific(fine-tuned) parameter를 최소화하고, 모든 pre-trained parameter를 조금만 바꿔서 down stream task를 학습한다.

위 방법 둘다 pre-train 과정에서 동일한 목적함수를 공유한다. 이때 일반적으로 language representation을 학습하기 위해 unidirectional language model을 사용한다.

<img width="586" alt="스크린샷 2024-03-13 오후 3 45 55" src="https://github.com/K-Saaan/papers/assets/111870436/de89550a-dbce-4a78-9fad-0bb0b23867ed">

[BERT / GPT / ELMo 비교]

<ELMo는 순방향 역방향 언어 모델을 모두 사용하지만 각각의 출력값을 concat해서 사용하기 때문에 양방향이 아닌 단방향 모델로 본다. 이것이 BERT에서 강조하는 deep bidirectional과의 차이점이다.>

본 논문에서 기존의 pre-trained representation 방법이 성능을 제한한다. 특히 fine-tuning 방식이 그렇다.
GPT의 경우 left-to-right 단방향 모델로 모든 토큰이 이전 토큰과의 attention만 계산해 문장 레벨 task에서는 차선책이 된다. QA와 같은 토큰 단위의 task에서는 context의 양방향을 포함하는 것이 중요한데 단방향 fine-tuning 방식은 성능이 떨어진다.
<최선책은 양쪽 토큰 모두의 attention을 계산하는 것이다.>

본 논문에 나오는 BERT는 앞서 언급한 비양방향 제약을 MLM(masked language model)을 pre-training 목적으로 사용하여 완화시켰다.

MLM은 입력 토큰의 일부를 랜덤하게 마스킹하고 해당 토큰이 구성하는 문장만을 기반으로 마스킹된 토큰의 원래 값을 예측하는 것이 목적이다.
left-to-right 모델과 다르게 MLM은 양방향 context를 융합해 deep bidirectional Transformer를 가능하게 했다. 추가로, MLM에서 text-pair representation으로 pre-train하면 "next sentence prediction"(다음 문장 예측) task적용할 수 있다.

### 2. Related Work
Related Work에서는 language representation을 pre-training하는 방법론에서 대표적인 것들을 리뷰하는 section이다.

#### 2.1 Unsupervised Feature-based Approaches
단어 representation 분야 연구는 수집년간 진행되어 왔다. 주요 method로는 non-neural method와 neural method로 구분된다.지속적인 연구를 통해 발전되어 왔을 만큼 word embedding의 pre-training은 오늘날 NLP 분야에서 중요한 부분이다.

word embedding을 통한 접근 방식은 크게 sentence embedding / paragraph embedding이 있다.

sentence representation 학습은 이전에 다음과 같은 방법을 사용했다.
1. rank candidate next sentence (다음 문장 후보들의 순위 메기기)
2. left-to-right generation (이전 문장이 주어졌을 때 다음 문장의 left-to-right generation 방법)
3. denoising auto-encoder (denoising auto-encoder에서 파생된 방법)

ELMo와 후속 모델들은 전통적인 word embedding 연구에서 left-to-tight, right-to-left 언어 모델을 사용해 context-sensitive feature들을 뽑아내는 방식으로 발전했다.

left-to-tight, right-to-left 두 표현 방식을 단순 concat 하는 것만으로 ELMo는 주요 NLP benchmarks(QA, 개체명 인식 등)에서 SOTA(State of the art 현재 최고 수준)를 달성했다. 하지만 이 역시 deep bidirection은 아니다.

#### 2.2 Unsupervised Fine-tuning Approaches
feature-based approaches의 초기 작업은 unlabeled text로 부터 word embedding parameter를 pre-train하는 방향으로 진행됐다.
최근에는 contextual token representation을 만드는 문장 또는 문서 인코더가 pre-train 되고 supervised downstream task에 맞춰 fine-tuning된다. 이런 방식은 scratch로 학습할 때 적은 parameter로 충분하다는 장점이 있다.
OpenAI GPT도 이런 방식으로 문장 단위 task에서 SOTA를 달성했다.

#### 2.3 Transfer Learning from Supervised Data
기계번역 자연어 추론과 같은 대규모 데이터셋에서 효과적인 전이학습(transfer learning)을 보여주는 연구도 있다.
전이학습은 CV에서도 중요성이 강조된다. ImageNet을 활용해 사전학습한 모델의 성능이 효과적이라고 한다.

### 3. BERT
본 논문에서 BERT는 pre-training / fine-tuning으로 나누어 설명한다.

pre-training task의 unlabeled data를 활용해 초기 파라미터를 설정한다. BERT 모델은 앞서 pre-trained parameter로 초기화한다. 그리고 모든 parameter를 downstream task의 labeling 된 데이터를 사용해 fine-tuning한다. 각 downstream task는 동일한 pre-trained parameter로 초기화 되어도 별도로 fine-tuning된 모델을 가진다.

pre-training
- 레이블링 하지 않은 데이터를 기반으로 학습

fine-tuning 
- 모델을 pre-training된 parameter로 초기화
- 모델을 레이블링된 데이터로 fine-tuning

실제 task에서 사용하는 모델은 초기에 동일한 parameter로 시작하지만 최종적으로 서로 다른 fine-tuning된 모델을 보유한다.
둘 사이의 구조적 차이는 거의 없다.

<img width="805" alt="스크린샷 2024-03-14 오후 4 35 31" src="https://github.com/K-Saaan/papers/assets/111870436/b31fcfaa-619f-4ce6-8619-34f58aeff5dd">

[Figure 1]

그림과 같이 pre-training의 parameter가 downstream task의 초기 값으로 사용되고 이후 fine-tuning과정에서 task에 맞게 조정된다.

BERT 모델 구조는 양방향 Transformer encoder를 여러 층으로 쌓은 multi-layer bidirectional Transformer encoder를 사용한다. 기존의 Transformer와 거의 유사한 구조로 되어 있다.
BERT는 BASE와 LARGE 두가지가 있다.
$$\large BERT_{BASE} (L=12, H=768, A=12, Total Parameters=110M) $$
$$\large BERT_{LARGE} (L=24, H=1024, A=16, Total Parameters=340M) $$

BASE의 경우 OpenAI GPT와의 비교를 위해 동일한 parameter를 사용했다. GPT는 토큰의 왼쪽 문맥만을 참조하지만 BERT는 양쪽 모두 참조할 수 있다.

- GPT는 다음 토큰을 맞추는 model을 만들기 위해 transformer와 decoder를 사용했고, BERT는 MLM(masked language model)과 NSP(next sentence prediction)를 위해 self-attention을 수행하는 encoder만 사용했다.

※ self-attention : bidirectional한 학습을 위한 것으로 한 단어와 다른 단어의 관계 정보를 처리하는 것.

BERT를 다룰 때 다양한 downstream task에서 잘 적용하기 위해 입력 표현이 애매하지 않게 하기 위해 하나 또는 한쌍의 sentence(문장 또는 인접한 텍스트들의 임의의 범위)를 하나의 토큰 시퀀스로 분명하게 표현해야한다.
시퀀스는 단일 또는 쌍으로 이루어진 문장을 말한다.
단어 임베딩은 WordPiece embedding을 사용한다.

모든 시퀀스의 첫번째 토큰은 항상 [CLS] 토큰이다. [CLS] 토큰과 대응되는 최종 hidden state는 분류 문제를 해결하기 위해 시퀀스 표현들을 종합한다. Input 시퀀스는 한 쌍의 문장으로 되어 있다. 두 문장을 하나의 시퀀스로 표현하기 위해 두 방법이 있다. 먼저 [SEP] 토큰으로 분류한다. 이후 각 문장이 A인지 B인지 구분하기 위해 구성하는 단어들을 임베딩(Segment Embedding)으로 표현한다. 

입력 토큰에서 token, segment와 해당 토큰의 position embedding을 더해서 input representation이 생성된다.
$$\large Input representation = segment + token + position $$

#### 3.1. Pre-training BERT
전통적인 방식인 left-to-right, right-to-left를 사용해 pre-train 하는 ELMo, GPT와 다르게 BERT는 2개의 unsupervised task를 이용해 학습한다.

##### T1. Masked LM
기존에는 left-to-right, right-to-left를 사용했지만, bidirectional을 사용하면 간접적으로 예측하려는 단어를 참조하게 되고 multi layer 구조에서 해당 단어를 예측할 수 있게 된다.
MLM은 input token의 일정 비율(15%)을 랜덤하게 마스킹하고 마스킹된 토큰을 예측한다. 하지만 [MASK]토큰은 pre-training에서만 사용되고 fine-tuning에서는 사용되지 않는다. 떄문에 pre-training과 fine-tuning 사이에 mismatch가 발생한다. 이를 해결하기 위해 다음과 같은 방식을 사용한다.
- 15%의 80%는 [MASK]토큰으로 바꾼다.                       ex) My name is BERT -> My name is [MASK]
- 10%는 랜덤 토큰(단어)로 바꾼다.                           ex) My name is BERT -> My name is man
- 10%는 바꾸지 않는다.(실제 관측 단어에 대한 표현을 bias하기 위함) ex) My name is BERT -> My name is BERT

이후 cross entropy loss를 사용해 원래의 토큰을 예측한다.

##### T2. Next Sentence Prediction(NSP)
QA, NLI와 같은 downstream task에서는 두 문장 사이의 관계를 이해하는 것이 중요하다.이러한 문장 간 관계는 language model로 알기는 어렵다. 이를 위해 BERT는 말뭉치에서 생성될 수 있는 NSP(next sentence prediction) task를 학습하기 위해 pre-train한다. 문장 A, B를 선택할 때 50%는 실제 A의 다음 문장인 B를(IsNext) 나머지 50%는 랜덤 문장 B를(NotNext) 선택한다.

ex) A : [CLS] I went to [MASK] [SEP=IsNext] B : I bought [MASK]
    A : [CLS] I went to [MASK] [SEP=NotNext] B : [MASK] is very good

<img width="805" alt="스크린샷 2024-03-14 오후 4 35 31" src="https://github.com/K-Saaan/papers/assets/111870436/b31fcfaa-619f-4ce6-8619-34f58aeff5dd">

[Figure 1]

그림에서 보듯 토큰 C는 NSP를 위해 사용된다. BERT는 이를 이용해 두 문장이 원래 붙어있었는지(IsNext) 아닌지를(NotNext) 학습한다.

##### Pre-training data
pre-training 과정에는 많은 데이터가 필요하다. BERT는 corpus 구축을 위해 BookCorpus(800M)와 English Wikipedia(2,500M)를 사용했다. Wikipedia는 리스트, 표, 헤더를 제외한 본문만 사용했다. 긴 인접 시퀀스를 뽑아내기 위해서는 문서 단위의 corpus를 사용하는 것이 문장 단위의 corpus를 사용하는 것보다 유리하다.

#### 3.2. Fine-tuning BERT
Transformer의 self-attention 메커니즘을 사용하면 BERT가 적절한 입력과 출력을 교환해 downstream task를 모델링할 수 있다.
BERT는 각 task에 따라 task specific 입출력을 받아서 각 task에 맞게 end to end로 parameter를 업데이트한다.
- Input
1. Sentence pairs in paraphrasing
2. Hypothesis-premise pairs in entailment
3. Question-passage paris in question answering
4. A degenerate text-none pair in text classification or sequence tagging
- Output
1. Token representation in sequence tagging or qusetion answering
2. [CLS] representation in classification(entailment or sentiment analysis)

대부분의 Fine-tuning task의 경우 TPU에서 1시간 GPU에서 몇시간 정도 걸린다.

### 4. Experiments
본 논문에서는 fine-tuning을 이용한 11개의 NLP task 결과를 보여준다.

#### 4.1. GLUE
<img width="979" alt="스크린샷 2024-03-15 오후 1 39 57" src="https://github.com/K-Saaan/papers/assets/111870436/b2d9d48d-0a46-46dc-8e62-ed9235cb0493">

[Table 1]

#### 4.2. SQuAD v1.1
<img width="375" alt="스크린샷 2024-03-15 오후 1 52 11" src="https://github.com/K-Saaan/papers/assets/111870436/9cf8aca8-2358-47e7-88fd-bb97ec9c7249">

[Table 2]

#### 4.3. SQuAD v2.0
<img width="369" alt="스크린샷 2024-03-15 오후 1 52 58" src="https://github.com/K-Saaan/papers/assets/111870436/c11b2b7b-e98a-4b14-90e8-a71a1e4699ad">

[Table 3]

#### 4.3. SWAG
<img width="380" alt="스크린샷 2024-03-15 오후 1 53 37" src="https://github.com/K-Saaan/papers/assets/111870436/ddcec046-9acb-4398-9a61-b04d56237be1">

[Table 4]

### 5. Ablation Studies
<img width="386" alt="스크린샷 2024-03-15 오후 1 54 22" src="https://github.com/K-Saaan/papers/assets/111870436/269e3be4-08bf-4685-a180-bb3fcc897cec">

[Table 5]

#### 5.1. Effect of Pre-training Tasks
앞서 언급한 방식들이 실제 성능에서 어떻게 보여지는지 확인할 수 있다.

- No NSP : MLM으로만 학습되고 NSP는 사용하지 않은 경우 QNLI, MNLI, SQuAD1.1애서 성능이 떨어진다.
- LTR & No NSP : 모든 task에서 No NSP 방식보다 성능이 떨어지고 특히 MRPC와 SQuAD에서 성능이 떨어지는 것을 볼 수 있다. bidirectional 요소의 제외 떄문인지를 확인하기 위해 초기화된 BiLSTM을 추가했을 때 SQuAD에서 성능이 향상된다.

#### 5.2. Effect of Model Size
모델의 크기가 fine-tuning 성능에 어떤 영향을 주는가?

<img width="365" alt="스크린샷 2024-03-15 오후 2 09 36" src="https://github.com/K-Saaan/papers/assets/111870436/aca04aa7-5f9a-43a2-a034-cf5783400bd4">

[Table 6]

모델의 크기가 클수록 성능이 좋은 것을 확인할 수 있다.

#### 5.3. Feature-based Approach with BERT
BERT를 기존의 방법대로 fine-tuning approach 했을 때와 ELMo 같은 feature-based approach로 사용했을 때
<img width="380" alt="스크린샷 2024-03-15 오후 2 19 30" src="https://github.com/K-Saaan/papers/assets/111870436/8cc17f55-c152-4912-94cd-f9de9d35a6ff">

[Table 7]

전체 layer를 가중합 하는 것 보다 마지막 4개의 layer를 concat하는 방법의 성능이 가장 좋았다. fine-tuning의 성능과 큰 차이가 없다.
