## [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Abstract
뛰어난 sequence transduction 모델은 encoder와 decoder가 포함된 recurrent나 convolution neural network의 복잡한 모델에 기초했다. 성능이 가장 좋은 모델은 encoder와 decoder를 합친 attention 매커니즘 모델이었다. 우리는 RNN과 CNN을 완전히 배제하고 오직 attention 매커니즘에 기초한 새롭고 간단한 네트워크 구조인 Transformer를 제안한다. 두 건의 기계 번역 작업에 대한 실험에서 이 모델은 병렬화가 가능하고 적은 학습시간을 필요로 하는 좋은 성능을 보여줬다. 우리는 Transformer가 다른 task에도 잘 일반화된다는 것을 크고 제한된 학습데이터를 가진 영어 구분분석에 적용함으로써 봤다.

-> 이전에는 encoder decoder와 RNN/CNN이 합쳐진 모델이 있었지만 attention 매커니즘을 사용하는 새로운 모델인 Transformer를 제안한다.

### 1. Introduction
RNN, LSTM, gated RNN은 모델링과 기계 번역 같은 sequence modeling과 transduction 문제에서 SOTA 방법으로 평가되었다. 이후에도 RNN과 encoder-decoder 아키텍처의 경계를 넓히기 위해 많은 노력이 있었다.
RNN 모델은 일반적으로 입력 및 출력 squence의 symbol position을 따라 계산한다. Step 위치에 따라 이전 hidden state ht-1과 position t가 input인 hidden state의 ht가 생성된다. 이러한 순차적 특성은 병렬처리를 막고 긴 시퀀스에서 메모리 제약으로 인해 예제간 batch가 제한되는 문제가 있다. 최근 작업에서 factorization trick과 conditional computation을 통해 많은 개선을 이뤘고 후자의 경우 모델 성능도 개선되었다. 하지만 sequential 연산에 대한 근본적인 문제는 남아있다.

-> Sequential 모델은 병렬 처리가 불가능하고(memory/computation에 부담) long-term dependency 문제가 있다.

### 2. Background
Extended Neural GPU, ByteNet 그리고 ConvS2S는 모두 sequential 연산을 줄이는 것을 목표로 했었다. 모두 CNN을 사용해서 input, output에 대해 병렬처리를 할 수 있었다. 이러한 모델에서는 임의의 두 입력 또는 출력 위치의 신호를 연결하는 데 필요한 작업 수가 위치 간 거리에 따라 ConvS2S의 경우 선형적으로, ByteNet의 경우 로그적으로 증가한다. 이로 인해 먼 위치 간의 종속성을 학습하기가 더 어려워진다. Transformer는 앞의 문제를 상수 연산으로 줄이고 가중치가 부여된 위치를 평균화하여 비용을 감소시켰다. 이는 Multi-Head Attention section에서 설명한다.
Self-attention은 sequence의 representation 연산을 위해 single sequence의 다른 위치를 연관시키는 attention 매커니즘이다. Self-attention은 독해, 요약, 텍스트 추론과 같은 다양한 작업에서 성공적으로 사용된다.
End-to-end memory networks는 sequence aligned recurrent 대신 recurrent attenion 매커니즘을 기반으로 해 simple-language QA나 language modeling task에서 좋은 성능을 보인다.
Transformer는 RNN이나 CNN을 사용하지 않고 입출력 연산을 위해 self-attention만을 사용하는 transduction model이다.


### 3. Model Architecture
대부분의 neural sequence transduction model은 encoder-decoder 구조를 가지고 있다. encoder는 input sequence of symbol representations (x1, .... , xn)을 연속된 rerpesentation z = (z1, ..., zn)에 매핑한다. z가 주어지면, decoder는 한 번에 하나씩 symbol의 출력 sequence(y1, ..., yn)을 생성한다. 각 step은 auto-regressive히며, 이전에 생성된 output을 다른 step의 input으로 사용한다.

-> 이전 단계 환료 후 다음 단계 수행할 수 있어 병렬처리가 불가능하다.

Transformer는 그림1처럼 여러겹의 self-attention과 point-wise로 된 완전 연결 encoder-decoder 구조를 따른다.

<img width="497" alt="스크린샷 2024-04-19 오후 3 23 50" src="https://github.com/K-Saaan/papers/assets/111870436/64f7962f-0250-4a78-82b8-8e29b71da895">

[Figure 1]

#### 3.1 Encoder and Decoder Stacks

##### Encoder
encoder는 n=6개의 동일한 layer stack으로 구성되어 있다. 각 layer에는 2개의 sub-layer가 있다. 첫번째는 multi-head self-attention 매커니즘이고 두번째는 Feed Forward network다. 우리는 두개의 sub layer에 각각 residual connection과 layer normalization을 사용한다. 각 sub layer의 output은 LayerNorm(x + Sublayer(x)), 이다. SubLayer(x)는 sub layer 자체에 구현된 함수(multi-head attention, feed forward)를 의미한다.
residual connection을 유용하게 하기 위해 모든 sub layer는 d_model=512 차원의 output을 생성한다.

- input / output의 d_model = 512
- multi-head attention과 feed forward 두개의 sub layer 존재
- residual connection, normalization 수행

##### Decoder
decoder 또한 n=6개의 동인한 layer stack으로 구성되어 있다. 각 endcoder sub layer 외에도 3번째 sub-layer를 추가하여 endcoder stack의 output에 대해 multi-head attention을 수행한다. encoder와 유사하게 각 sub layer에 residual connection과 layer normalization을 사용한다. 또한 decoder stack의 self-attention sub layer를 수정해 position이 subsequence position에 attending 하는 것을 막는다.(Masked Multi-Head Attention) masking은 output embedding이 한 위치의 offset이라는 사실과 결합해 position i에 대한 예측이 i보다 적은 수의 위치에서 알려진 output에만 의존할 수 있도록 한다.

- masked multi-head attention,  multi-head attention, feed forward 세개의 sub layer 존재
- residual connection, normalization 수행
- 두번째 sub layer에서 encoder의 출력을 k,v로 사용
- 참조를 막기 위해 masking 사용

#### 3.2 Attention
attention은 query와 key-value 쌍을 q, k, v 그리고 output이 모두 vector인 output에 매핑하는 것으로 묘사된다. output은 values의 weight sum으로 계산되며, 각 value에 할당된 weight는 query와 해당 key의 compatibility function에 의해 계산된다.

- Query : 영향을 받는 벡터
- Key : 영향을 주는 벡터 
- Value : Key의 가중치 벡터 
- Query가 정답일 때 정답과 가장 비슷한 Key를 선택하고 해당 Key의 value값을 가져온다(attention)

##### 3.2.1 Scaled Dot-Product Attention
<img width="219" alt="스크린샷 2024-04-20 오후 4 50 15" src="https://github.com/K-Saaan/papers/assets/111870436/3fe0cbd8-f56e-4817-8760-6717505f22db">

[Figure 2]

그림2를 Scaled Dot-Product Attention이라고 부른다. key와 dot-product를 계산하고 input은 d_k 차원의 query와 key 그리고 d_v차원의 value로 구성된다.
우리는 모든 key와 query의 dot product를 계산하고, 각 값을 root(d_k)로 나눈 후, softmax 함수를 적용해 value에 대한 weight를 구한다.
우리는 일련의 queries에 대해 동시에 attention 함수를 계산하여 matrix Q로 묶는다. keys와 values도 K, V로 묶는다. 계산식은 다음과 같다.

<img width="317" alt="스크린샷 2024-04-22 오후 3 35 48" src="https://github.com/K-Saaan/papers/assets/111870436/a1e8eb7a-d2b5-448d-974a-75007bbf41ea">

[식 1]

주로 사용되는 attention function은 additive attention과 dot-product(multiplicative)attention 이다. Dot-product attention은 scaling factor인 1/root(d_k)를 제외하고 동일하다. Additive attention은 single hidden layer가 있는 FFN을 사용하여 composibility function을 계산한다. 두 방법은 이론적으로 복잡도가 비슷하며, dot-product attention은 matrix를 통해 최적화된 연산을 구현할 수 있어 더 빠르고 공간 효율적이다.
d_k의 값이 작을 때 dot-product와 scaled dot-product가 유사하게 수행되지만, d_k의 값이 큰 경우 additive attention의 성능이 더 좋다.
우리는 d_k 값이 큰 경우 dot-product가 커져 softmax 함수가 매우 작은 기울기를 가지게 된다는 것을 알았다. 이를 방지하기 위해 dot-product를 1/root(d_k)로 scaling 해준다.

- Query와 Key를 dot-product하여 attention score를 계산 = 1
- scaling 및 softmax 해준다 = softmax(1/root(d_k)) = 2
- 2 * Values를 통해 각 토큰별 attention을 계산한다.
-> 토큰별 attention을 확인하기 위한 과정

##### 3.2.2 Multi-Head Attention
우리는 d_model 차원의 q, k, v를 사용하여 single attention을 수행하는 대신 각 d_k, d_k, d_v차원에 대해 학습된 서로다른 linear projection을 사용하여 q, k, v를 h회 linear projection할 때 성능이 좋은 것을 발견했다. 각 queries, keys, values의 projected voersion에서 attention 함수는 병렬로 수행되며, d_v차원 output을 생성하고 이를 concat하여 다시 d_model 차원의 output을 생성한다.
Multi-head attention을 통해 다른 Position의 서로 다른 representation subspaces의 정보에 공통으로 attention 할 수 있다.

<img width="414" alt="스크린샷 2024-04-22 오후 4 03 58" src="https://github.com/K-Saaan/papers/assets/111870436/90b9c286-fa0d-49c7-83bf-633e86e46827">

본 작업에서는 h=8개의 병렬 attention layer 또는 head를 사용한다. 각 head의 축소된 차원으로 인해 총계산 비용은 전체 차원으로 갖는 single head attention 비용과 유사하다.

##### 3.2.3 Applications of Attention in our Model
Transformer에서 multi-head attention을 사용하는 세가지 방법이 있다.

encoder-decoder attention : queries는 이전의 decoder layer에서 그리고 keys와 values는 encoder의 output에서 온다. decoder의 모든 position이 input sequence의 모든 position에 참조될 수 있다. sequence-to-sequence 모델에서 일반적인 encoder-decoder attention 매커니즘과 동일하다.

encoder contains self-attention layer : self-attention layer의 모든 key, value, query는 동일한 encoder의 이전 layer output에서 온다. encoder의 각 position은 이전 layer의 모든 position에 attend할 수 있다.

마찬가지로, decoder의 self-attention layer는 decoder의 각 position이 해당 위치까지 decoder의 모든 position에 attend하도록 한다. 우리는 auto-regressive property를 유지하기 위해 decoder에서 leftward 정보를 참조하는 것을 막아야 한다. softmax 결과가 0에 수렴하도록 해 masking을 수행한다.


#### 3.3 Position-wise Feed-Forward Networks
각 encoder와 decoder layer는 fully connected feed-forward network를 포함하고 있으며, 각 position에 동일하게 적용된다. FFN은 사이에 ReLU 활성함수가 있는 두 linear transformation으로 이루어져 있다.

<img width="273" alt="스크린샷 2024-04-22 오후 4 44 39" src="https://github.com/K-Saaan/papers/assets/111870436/9c0a61b0-6a99-4b74-9381-f6ed3cec150a">

[식 2]

linear transformation은 여러 position마다 동일하고 layer마다 다른 Parameter를 가진다. 

#### 3.4 Embedding and Softmax
다른 sequence transduction 모델과 유사하게, learned embedding을 사용해 input/output token을 d_model 차원의 벡터로 변환한다. 또한 학습된 linear transformation과 softmax 함수를 사용해 decoder output을 예측된 다음 token 확률로 변환한다. 본 모델에서 embedding layer와 pre-softmax linear transformation 사이에서 동일한 weight matrix를 공유한다. embedding layer에서 우리는 가중치에 root(d_model) 곱을 한다.

#### 3.5 Positional Encoding
본 모델은 recurrence와 convolution 모델을 포함하지 않기 때문에 모델이 sequence의 순서를 사용하려면 sequence에서 token의 상대적 또는 절대적 position에 정보를 추가해줘야한다. 이를 위해 우리는 encoder와 decoder stack의 바닥에 있는 input embedding에 'positional encoding'을 추가한다. positional encoding은 embedding과 동일한 차원 d_model을 가진다. 본 연구에서는 sine, cosine 함수를 사용한다.

<img width="286" alt="스크린샷 2024-04-22 오후 5 07 13" src="https://github.com/K-Saaan/papers/assets/111870436/fbc7af47-af69-4f14-8a88-fc1996a2bc68">

pos는 position(token의 위치), is는 차원을 의미한다. 각 positional encoding의 각 차원은 sinusoid에 해당한다. 파장은 2π에서 10000 · 2π까지 기하학적 수열을 형성한다. PE_pos+k가 PE_pos의 선형 함수로 표현될 수 있어 모델이 상대적 position을 쉽게 학습할 수 있을 것으로 가정했기 때문에 이 함수를 사용한다.
학습된 positional embedding을 사용하여 실험한 결과 거의 동일한 결과를 생성하는 것으로 나타났습니다. 더 긴 sequence에서도 추론이 가능한 sinusoidal에서도 사용할 수 있다.

### 4. Why Self-Attention
본 섹션에서는 hidden layer안에 대표적인 sequence transduction인 encoder나 decoder와 같이 symbol representations의 one variable-length sequence를 같은 길이로 mappingg하는 방법을 사용해 self-attention layer와 recurrent, convolutuin layer와의 비교를 한다. self-attention이 필요하다고 생각한 동기는 3가지이다.

1. layer별 총 연산의 복잡성.
2. 필요한 최소 sequential 작업으로 측정된 병렬처리 연산량.
3. Network에서 long-range dependencies 사이 path length. long-range dependencies 학습은 많은 번역 task에서 중요하다. dependency를 학습하는 것에 영향을 미치는 하나의 중요한 요소는 전달해야하는 forward, backward signal의 길이이다. input, output 위치의 길이가 짧을수록 학습은 쉬워진다. 그래서 서로 다른 layer type로 구성된 네트워크에서 input과 output위치 사이 길이가 Maximum 길이를 비교한다.

<img width="666" alt="스크린샷 2024-04-23 오후 1 52 34" src="https://github.com/K-Saaan/papers/assets/111870436/586eb67f-5768-4ed5-8133-a0478c680751">

[Table 1]

Table1에서 알 수 있듯 self-attention layer는 순차적으로 실행되는 일정한 수의 작업으로 모든 위치의 layer를 연결하는 반면, recurrent layer는 O(n)개의 순차적 작업이 필요하다. 계산 복잡도 측면에서, 시퀀스 길이 n이 표현 차원 d보다 작을 때 self-attention layer는 recurrent layer보다 빠르다. 매우 긴 시퀀스와 관련된 작업의 계산 성능을 향상시키기 위해 self-attention은 각 출력 위치를 중심으로 한 입력 시퀀스에서 크기 r의 이웃만 고려하도록 제한될 수 있다.
커널 너비가 k<n인 단일 convolution layer는 모든 input 및 output 출력 위치 쌍을 연결하지 않는다. network에서 convolution layer는 일반적으로 recurrent layer보다 k배 비싸다. 하지만 convolution은 복잡성을 줄이고 self-attention과 유사하다.
추가적으로 self-attention은 해석 가능한 모델을 더 생성할 수 있다. 개별 attention-head는 다양한 작업을 수행하는 방법을 명확하게 학습하고 많은 사람들이 문장의 구문 및 의미 구조와 관련된 행동을 보이는 것으로 보인다.

### 5. Training

#### 5.1 Training Data and Batching
우리는 약 450만 개의 문장 쌍으로 구성된 표준 WMT 2014 영어-독일어 dataset을 학습했다. 문장은 byte-pair encoding을 통해 인코딩되었고, 약 37,000 토큰의 공유 sourcetarget 어휘를 가진다. 영어-불어는 3.6백만개의 문장으로 구성된 더 큰 WMT 2014 영어-불어 데이터셋을 사용하고 token을 3.2만개의 단어 단위로 분할한다. 문장 쌍을 대략적 sequence 길이로 묶여있다. 각 훈련 batch는 약 2.5만개의 소스 token과 2.5만개의 target token이 담긴 문장 쌍을 포함한다.

#### 5.2 Hardware and Schedule
NVIDIA P100 GPU 8개를 가진 기계에서 우리의 모델을 학습시켰다. 논문에서 묘사된 hyperparameter를 사용한 기본 모델의 경우 각 학습 step은 0.4초가 소요되었다. 기본 모델을 총 100,000 steps(12시간) 동안 학습했다. large 모델의 경우 각 Step 학습 시간은 1.0초 였다. 총 300,000 steps(3.5일)간 학습했다.


#### 5.3 Optimizer
<img width="475" alt="스크린샷 2024-04-23 오후 2 24 39" src="https://github.com/K-Saaan/papers/assets/111870436/a55b54bd-430d-4a65-9de8-336def729b83">

[식 3]

#### 5.4 Regularization
##### Residual Dropout
- 각 sub-layer 입력에 추가되고 정규화되기 전에 각 sub-layer 출력에 dropout을 적용한다. 또한 encoder-decoder stack 모두에서 embedding과 positional encoding의 합에 dropout을 적용한다. 기본 모델의 경우 Pdrop=0.1의 비율을 사용한다.

##### Label Smoothing

### 6. Result

#### 6.1 Machine Translation
<img width="671" alt="스크린샷 2024-04-23 오후 2 33 10" src="https://github.com/K-Saaan/papers/assets/111870436/0ba4bbc1-465a-4f81-b3fc-f807fc268ff1">

[Table 2]

#### 6.2 Model Variations
<img width="671" alt="스크린샷 2024-04-23 오후 2 33 17" src="https://github.com/K-Saaan/papers/assets/111870436/814de263-cf46-4ae6-b1fb-914e56ed2829">

[Table 3]

#### 6.3 English Consituency Parsing
<img width="671" alt="스크린샷 2024-04-23 오후 2 33 22" src="https://github.com/K-Saaan/papers/assets/111870436/5d64d8cc-855b-4652-8d93-8f354c28f3d2">

[Table 4]

### 7. Conclusion
1. Recurrent layer를 Multi-head self-attention으로 대체하여 attention만을 사용하는 transformer를 제안했다.
2. Translation task는 기존의 RNN, CNN 모델 보다 더 좋은 성능을 보여주며, WMT 2014 English-to-French translation task에서 SOTA를 달성했다.
3. Transformer는 텍스트, 이미지, 오디오, 영상과 같이 상대적으로 큰 input, output을 필요로하는 작업들을 효율적으로 처리하기 위해 사용될 수 있다.












