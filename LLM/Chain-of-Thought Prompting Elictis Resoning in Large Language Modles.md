# Chain-of-Thought Prompting Elictis Resoning in Large Language Models

제목 : Chain-of-Thought Prompting Elictis Resoning in Large Language Models

학회 : **NeurIPS 2022**

연도 : 2022.01.28

저자 : [Jason Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei,+J), [Xuezhi Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+X), [Dale Schuurmans](https://arxiv.org/search/cs?searchtype=author&query=Schuurmans,+D), [Maarten Bosma](https://arxiv.org/search/cs?searchtype=author&query=Bosma,+M), [Brian Ichter](https://arxiv.org/search/cs?searchtype=author&query=Ichter,+B), [Fei Xia](https://arxiv.org/search/cs?searchtype=author&query=Xia,+F), [Ed Chi](https://arxiv.org/search/cs?searchtype=author&query=Chi,+E), [Quoc Le](https://arxiv.org/search/cs?searchtype=author&query=Le,+Q), [Denny Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou,+D)

# 1. Introduction

언어 모델의 크기를 키우면 성능이 향상된다. 하지만, 크기를 키우는 것 만으로는 산술, 상식, 상징적 추론과 같은 과제에서 높은 성능을 달성하기 어렵다.

본 연구는 LLM의 추론 능력을 쉬운 방법으로 활용할 수 있는지 탐구하고자 하며, 두가지 아이디어에 영감을 받았다.

<aside>
💡

산술 추론은 최종 답변에 이르는 논리 단계를 통해 성능 향상을 할 수 있다. 

이전 연구(Program induction by rationale generation Ling et al.. 2017)에서 모델이 중간 논리 단계를 생성할 수 있도록 사전학습 또는 미세조정을 통해 학습시키는 방법을 사용했고 추가로 자연어 대신 형식 언어를 사용하는 neuro-symbolic 방법도 활용됐다.

<aside>
❓

형식 언어를 사용하는 neuro-symbolic 방법

자연어 대신 수학적 기호나 논리 표현과 같은 형식 언어를 사용하여 중간 추론 과제를 생성

Ex) 10 - 3 = 7 처럼 수학적 표기법을 사용해 명확하게 표현

형식 언어는 명확하고 정밀한 계산을 통해 답을 도출

자연어는 직관적이고 사람이 이해하기 쉬움

</aside>

</aside>

<aside>
💡

LLM은 prompt를 통해 몇 가지 예제를 받아 과제를 학습하는 in-context learning을 할 수 있다.

이 방법은 새로운 과제마다 미세조정 없이 입-출력 예제를 prompt에 넣어 모델이 과제를 학습하도록 하고 간단한, 질문-답변 과제를 성공적으로 수행한다.

</aside>

하지만 위 두 방식에는 한계가 있다.

미세조정 학습을 위한 고품질의 데이터를 생성(중간 논리를 생성하는)하는 것은 많은 비용이 발생하고 전통적인 few-shot prompting 방법은 추론 능력을 요구하는 과제에서 성능이 저조하다. 

본 연구는 이러한 한계를 극복하고 각 방법의 장점을 결합한 새로운 방식을 제안

언어모델이 <input, chain of thought, output>으로 구성된 prompt를 통해 추론 과제에서 few-shot prompting을 수행할 수 있도록 한다.

[chain of thought 이하 CoT]

→ 프롬프트만 사용하는 것은 미세조정과 같이 대규모 학습 데이터셋을 필요로 하지 않으며, 단일 모델이 일반성을 잃지 않고 여러 과제를 수행할 수 있도록 한다. 

# 2. Chain-of-Thought Prompting

여러 단계에 따라 풀어야 하는 수학 문제 같이 복잡한 추론 과제를 풀기 위해서는 문제를 단계별로 나누고 각각을 해결하면서 최종 답을 내는 것이 일반적이다.

본 연구는 모델이 문제의 최종 답을 도출하기 위한 중간 추론 단계인 CoT를 생성할 수 있도록 하는 것이다. 실제로 CoT 예제를 제공했을 때 언어모델이 CoT를 생성하는 것을 볼 수 있었다.

그림1은 모델이 문제를 해결하기 위해 CoT를 생성한 예제이다. 모델이 생성한 CoT에 최종 답에 이르기까지의 과정이 나타난 것을 볼 수 있다.

<img width="880" alt="Image" src="https://github.com/user-attachments/assets/6b9cc9e0-526e-4e55-95b6-4dc1afbc3eec" />

CoT prompting의 특징

1. CoT는 원칙적으로 모델이 여러단계로 구성된 문제를 단계별로 나눌 수 있게 하며, 이는 더 많은 추론 단계가 필요한 문제에 추가 계산을 할당할 수 있도록 한다.
2. CoT는 모델이 특정 답에 도달한 과정에 대해 확인할 수 있도록 해, 추론 과정 중 잘못된 부분에 대해 디버깅할 수 있는 기회를 제공한다. (다만, 모델의 계산을 완전히 특성화하는 것은 여전히 미해결이다.)
3. 수학 문제, 상식 추론, 상징적 추론 등 다양한 작업에 적용할 수 있으며, 인간이 언어를 통해 해결할 수 있는 모든 작업에 적용 할 수 있다.
4. CoT는 몇 가지 예제를 prompt에 포함시키면 큰 언어 모델에서 정답을 쉽게 유도할 수 있다.

# 3. Arithmetic Reasoning

그림 1에 나타난 문제와 같은 산술 추론은 인간에게는 쉽지만, 언어 모델에게는 어려운 작업이다. 

본 연구에서는 여러 모델과 벤치마크로 실험을 진행했다.

### Benchmarks

1. GSM8K : 수학 문제 벤치마크
2. SVAMP : 다양한 구조를 가진 수학 문제 데이터셋
3. ASDiv : 다양한 수학 문제 데이터셋
4. AQuA : 대수적 문제 데이터셋
5. MAWPS : 수학 문제 벤치마크

### Language models

1. GPT-3(InstructGPT)
    
    350M, 1.3B, 6.7B, 175B
    
2. LaMDA
    
    422M, 2B, 8B, 68B, 137B
    
3. PaLM
    
    8B, 62B, 540B
    
4. UL2 
20B
5. CODEX

### Standard prompting

성능 기준선으로는 Brown et al.(2020)이 제안한 few-shot prmpting을 사용했다.

### Chain-of-thought prompting

본 연구에서는 few-shot prompting의 각 예제에 답과 관련된 CoT를 추가하고자 했다. 

![Image](https://github.com/user-attachments/assets/ce357dd0-90b2-4e25-ba4e-5a4474670274)

### Results

CoT prompting이 수학 문제 전반에 올바른 추론을 유도할 수 있는지 확인하기 위해 standard prompting과  비교했다.

<img width="889" alt="Image" src="https://github.com/user-attachments/assets/7813bb90-9b69-4b3a-93ed-0dcb753778a4" />

1. 모델 크기의 영향

그림4에서 모델의 크기가 충분히 클 때 CoT의 성능 향상이 두드러지는 것을 볼 수 있다(논문에서는 emergent ability라고 표현). 크기가 작은 모델에서는 CoT가 효과적이지 않고 약 100B 매개변수 이상 규모의 모델에서만 성능 향상이 나타난다.

![Image](https://github.com/user-attachments/assets/d4de8eab-2bf3-438e-a9b4-8c327b0241d3)


2. 복잡한 문제에서 더 큰 성능 향상

CoT prompting은 더 복잡한 문제일수록 큰 성능 향상을 보인다. GSM8K에서 가장 어려운 문제에서 성능이 두 배 이상 향상되었지만, MAWPS의 가장 쉬운 문제에서는 성능이 하락하거나 미미했다. (부록 표3)

<img width="866" alt="Image" src="https://github.com/user-attachments/assets/d3941215-7d30-419a-9cbe-9d6e7352afc7" />

3. CoT은 이전 방법과 비교했을 때 우수

GPT-3 175B와 PaLM 540B의 경우 미세조정하는 방법과 비교했을 때도 CoT의 성능이 우수한 것을 확인했다.

그림4는 CoT를 활용했을 때 기준선을 뛰어넘는 성능을 보이는 것을 보여준다. 

<aside>
💡

CoT prompting 효과 분석

CoT prompting 효과적인가?

LaMDA 137B 모델로 GSM8K 벤치마크를 수행 후 올바른 답을 반환한 50개를 샘플링하여 분석했다. 생성된 CoT는 논리 및 수학적으로 정확했으나 두 개는 정답에 우연히 도달했다. 

잘못된 답을 반환한 50개 분석 결과, 46%는 사소한 오류(계산기 오차, 기호 매핑 오류, 추론 단계 누락)를 제외하면 거의 정확했다. 나머지 54%는 의미적 이해 또는 일관성에 큰 오류를 보였다.

→ CoT prompting이 추론에 기여했고 prompt가 개선될 경우 사소한 오류 같은 문제가 개선될 수 있을 것으로 보임

모델 크기에 따른 비교

PaLM 62B를 사용했을 때 발생한 오류가 PaLM 540B로 확장했을 때 개선되는지 분석한 결과, 모델 크기를 키웠을 때 많은 오류가 해결된 것을 확인 할 수 있었다.

</aside>

CoT prompting의 변형

1. Eqation Only
CoT의 효과가 중간 추론 과정 때문인지 확인하기 위해  모델이 답을 반환하기 전에 수학적 수식만 출력하도록 prompt를 구성해서 수식만으로 충분한지 확인하는 실험을 진행.
결과 : 해당 방법이 GSM8K에서는 눈에 띄는 성과가 없음 → GSM8K의 질문이 추론 없이 바로 수식으로 변환하기에 너무 어렵기 때문. 하지만, 한두 단계로 해결 가능한 문제의 경우 본 방법이 성능 향상에 영향을 미침

<img width="319" alt="Image" src="https://github.com/user-attachments/assets/4123dce9-16f8-438e-946a-164e0f5e7300" />


2. Variable Compute Only
추론 단계를 명확히 작성하는 것이 중요한가?
CoT prompting이 중간 추론 단계를 제공하기 때문에 표준 prompting 방식 보다 많은 연산을 사용하게 해서 성능이 향상된건가?
 Variable Compute 효과를 확인하기 위해 자연어로 제공한 중간 추론 단계를 모두(”.”)으로 출력하도록 했다.
결과 : 기준선과 비슷한 성능을 보였으며, 이 실험을 통해 variable Compute Only로는 성능 향상을 재현하기 어려움 
3. Chain of Thought After Answer
 CoT prompting이 실제로 답을 도출하는 데 중요한지, 모델이 사전 학습 중 얻은 지식을 활성화시키는 역할을 하는지 알아보기 위해 기존 CoT prompting 방식을 변경해 CoT prompt를 답 뒤에 위치하도록 설정했다. 기본 prompting과 비슷한 성능을 보였고 이걸 통해 CoT prompting이 추론 과정에서 중요한 역할을 한다는 것을 확인했다.

### Robustness of Chain of Thought

CoT prompting이 다양한 CoT 예제에 대해서도 성능이 robust한지 확인하기 위한 실험을 진행했다.

- 다양한 작성자에 의해 작성된 예시
- 다양한 예제 세트
- 다양한 언어 모델 종류와 크기
- 예제 순서 및 수

예제에 따라 조금씩 다른 성능을 보였지만 모두 기본 prompting보다 높은 성능을 보였다. 

CoT prompting이 특정 형식에 한정되지 않음을 시사한다.


<img width="362" alt="Image" src="https://github.com/user-attachments/assets/9a04fc19-1005-4c8a-801c-54e8b636af12" />

# 4. Commonsense Reasoning

CoT prompting이 광범위한 상식 추론 문제에도 적용 가능한지 확인하기 위한 실험.

**Benchmarks**

1. CSQA : 복잡한 의미를 포함하며 종종 사전 지식을 요구하는 세계에 대한 상식 질문
2. StrategyQA : 질문에 답하기 위해 여러 추론 단계를 거쳐야 하는 문제
3. BIG-bench
    1. Date Understanding : 맥락에서 날짜를 추론하는 문제
    2. Sports Understanding : 스포츠와 관련된 문장이 타당한지 판단
4. Date Understanding : 맥락에서 날짜를 추론하는 문제
5. Sports Understanding : 스포츠와 관련된 문장이 타당한지 판단
6. SayCan : 로봇에게 지시하는 자연어 명령(물리적 환경과 로봇의 상태에 따라 실행 가능한지 평가)


<img width="862" alt="Image" src="https://github.com/user-attachments/assets/70cb05ea-c43d-48e0-8bfc-5141df881345" />


# 5. Symbolic Reasoning

symbolic reasoning은 인간에게 간단하지만, 언어 모델에게는 도전적인 과제이다. CoT prompting으로 symbolic reasoning에서 성능을 향상시킬 수 있는지 확인한다.

과제

1. Last Letter Concatenation
이름 각 단어의 마지막 문자를 연결
Ex) Amy Brown → yn
2. Coin Flip
동전을 뒤집거나 뒤집지 않을 때, 동전이 앞면인지 뒷면인지 판단
Ex) 앞면인 동전을 뒤집었을 때 동전이 앞면인가? → 아니다

데이터셋 구성

3. In-Domain
train/few-shot 예제와 동일한 구조의 테스트셋을 사용
Ex) 두 단어 이름의 마지막 문자 연결
4. Out-of-Domain(OOD)
train/few-shot 예제보다 더 많은 수를 포함한 데이터셋 사용 
Ex) 세 단어 이름의 마지막 문자 연결

In-Domain의 경우 표준 prompting과 CoT prompting 모두 성능 향상을 보였고 CoT를 적용한 경우 거의 100%의 성능을 보임. 

OOD에 표준 prompting을 적용했을 때 유의미한 성능 향상을 보이지 않음

CoT prompting을 적용했을 때 성능 향상을 보였고 모델 크기가 커질수록 성능이 큰 폭으로 향상됨 

(단, In-Domain의 성능이 더 높음)

**결론**

Symbolic Reasoning은 단계적으로 생각하면서 규칙을 적용해야하는 작업이기 때문에 OOD 과제의 경우 모델이 이전에 보지 못해 새롭게 학습해야 한다. CoT는 이러한 규칙을 단계별로 진행하면서 적용할 수 있도록 도와준다. 이를 통해 복잡한 문제 해결에 도움을 주는 것을 확인했다.


<img width="358" alt="Image" src="https://github.com/user-attachments/assets/ebc10a3f-630c-4554-a85b-bbafd182a025" />

In-Domain의 경우 표준 prompting과 CoT prompting 모두 성능 향상을 보였고 CoT를 적용한 경우 거의 100%의 성능을 보임. 

OOD에 표준 prompting을 적용했을 때 유의미한 성능 향상을 보이지 않음

CoT prompting을 적용했을 때 성능 향상을 보였고 모델 크기가 커질수록 성능이 큰 폭으로 향상됨 

(단, In-Domain의 성능이 더 높음)

**결론**

Symbolic Reasoning은 단계적으로 생각하면서 규칙을 적용해야하는 작업이기 때문에 OOD 과제의 경우 모델이 이전에 보지 못해 새롭게 학습해야 한다. CoT는 이러한 규칙을 단계별로 진행하면서 적용할 수 있도록 도와준다. 이를 통해 복잡한 문제 해결에 도움을 주는 것을 확인했다.

<img width="358" alt="Image" src="https://github.com/user-attachments/assets/d2411b42-47ae-408c-b972-129c5294424c" />


In-Domain의 경우 표준 prompting과 CoT prompting 모두 성능 향상을 보였고 CoT를 적용한 경우 거의 100%의 성능을 보임. 

OOD에 표준 prompting을 적용했을 때 유의미한 성능 향상을 보이지 않음

CoT prompting을 적용했을 때 성능 향상을 보였고 모델 크기가 커질수록 성능이 큰 폭으로 향상됨 

(단, In-Domain의 성능이 더 높음)


# 6. Disscusion

- Arithmetic Reasoning에서 큰 성능 향상을 보였다
- Commonsense Reasoning에도 CoT가 적용 가능한 것을 확인
- Symbolic Reasoning에서 In-Domain, OOD 모두에서 CoT prompting이 효과적인 것을 확인.

→ 모든 실험에서 기존 언어 모델을 미세조정 없이 prompting만으로 성능을 향상시켰다.

CoT prompting의 특성

- 모델 규모가 커질수록 성능이 크게 증가한다.

한계

- 인간의 사고 과정을 모방하지만, 모델이 실제로 추론을 하는지는 미해결
- few-shot에서는 CoT prompt를 작성하는 비용이 낮지만 대규모 데이터셋에서는 비용이 높을 수 있음
- CoT가 항상 올바른 추론 과정을 보장하지는 않고 잘못된 답변을 초래할 수 있다. 언어 모델의 factual accuracy를 개선하는 연구가 필요하다
- CoT를 활용한 성능 향상이 대규모 모델에서만 나타나 실제 응용에서는 비용이 높을 수 있어 작은 모델에서도 활용할 수 있는 연구가 필요하다

<aside>
💡

factual accuracy

언어 모델이 생성한 응답이 사실적이고 신뢰할 수 있는가를 평가

</aside>

factual accuracy

언어 모델이 생성한 응답이 사실적이고 신뢰할 수 있는가를 평가

# 7. Related Work

1. 중간 단계를 활용한 추론 문제 해결
- Ling et al.(2017) 수학 문제를 해결하기 위해 자연어로 된 중간 단계를 생성하는 방식을 처음으로 제안. formal languages를 사용한 추론 방법과 대조되는 접근법
- Cobbe et al.(2021)은 Ling의 연구를 확장하여 더 큰 데이터셋을 생성해 사전학습된 모델을 미세조정했다.
- Nye et al.(2021)은 Python 프로그램의 최종 출력을 예측하는 것이 아닌 각 줄의 중간 결과를 단계적으로 예측하도록 했다. → 최종 출력을 직접 예측하는 것보다 더 나은 성능을 보임.
2. Prompting 연구와의 연관성
- few-shot prompting을 통해 훈련 없이 새로운 과제를 학습할 수 있다는 가능성을 제시
- prompting으로 성능을 향상시키기 위한 다양한 연구가 진행됨
3. CoT와 2번의 차이
- 기존 연구는 주로 prompt 입력 부분을 개선하거나 보강하는 데 초점이 맞춰짐.
- CoT prompting은 추론 단계가 있는 중간 단계를 통한 문제 해결 방법을 제시하여 모델이 복잡한 추론 작업을 더 잘 수행할 수 있도록 도와줌

# 8. Conclusions

CoT prompting 방법이 언어 모델에서 추론 능력을 향상시키고 다양한 작업에 적용 가능한지 탐구했다. 본 연구에서 알게된 사실은 다음과 같다.

1. 산술, 상식, 상징적 추론 과제에서 실험한 결과 CoT prompting이 성능 향상에 영향을 주는 것을 확인
2. 충분히 큰 모델에서 효과적인 성능 향상을 보인다.
3. 모델의 크기가 클수록 더욱 큰 성능 향상을 보인다.