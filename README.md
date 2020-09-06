# Valence-Arousal_based_sentiment_analysis

해당 프로젝트는 양재 AI 허브에서 진행된 AI Collage 과정의 팀 프로젝트에 앞선 사전 시도로, Valence-Arousal Model을 기반으로 다중감성분석을 시도하였다.  

<br>

## 1. 프로젝트 개요

AI를 활용한 자연어 처리에서 감성분석은 소비자의 선호를 분석하는 수준에서 사회적인 트렌드를 분석하는 수준까지 확장되며 사용되고 있다. 다만 그 분류는 여전히 긍정과 부정에 그치는 수준으로, 얼굴 표정을 통해 보다 다양한 정서에 대한 분류를 시도하고 있는 영상 처리 분야에 비하면 여전히 부족한 수준이라고 느껴진다.  
이에 비해 사용자들은 챗봇이나 AI 음성비서 서비스에 정서적인 기대를 품곤 한다. 이아라 외(2019)<sup>[[1]](#footnote_1)</sup>의 상담 챗봇을 활용한 연구에 따르면 사람들은 비밀보장과 익명성에 대한 신뢰감을 바탕으로 프로그램과의 대화과정을 보다 편하게 느끼는 것으로 보고되었다. 또한 한국소비자원의 조사<sup>[[2]](#footnote_2)</sup>에 따르면 구매 이전 AI 스피커에 기대한 특성으로 23%가 기기와의 일상 대화를 꼽았으며, 불만 사례 중 기기와 이용자 간의 자연스러운 연결형 대화 곤란이 45%로 보고되었다. 나아가 박지혜(2018)<sup>[[3]](#footnote_3)</sup>의 연구에 따르면 이러한 불편사항은 인공지능 스피커의 의인화를 통해 일정부분 해소 가능하다는 점을 알 수 있다. 이러한 연구들은 자연어 처리 분야에서 다중감성분석의 수요와 필요성을 보여준다.  

해당 프로젝트에서는 Valence-Arousal이라는 정서 모델을 바탕으로 다양한 정서에 대한 분류를 시도하였고, 그 과정과 결과는 다음과 같다.  

> #### Reference
> <a name="footnote_1">[1]</a> : 이아라, 김효창, 차민철, 지용구 (2019). 상담 이론 기반의 심리 상담 챗봇을 활용한 내담자 경험 연구. 대한인간공학회지, 38(3), 161-175  
> <a name="footnote_2">[2]</a> : 강민영(2017), 인공지능(AI) 가전제품 문제점 및 개선방안(음성인식 스피커를 중심으로), 한국소비자원 시장조사국 거래조사팀 조사보고서  
> <a name="footnote_3">[3]</a> : 박지혜, 주재우(2018), 인공지능 스피커의 지속적 사용의도를 높이는 행동경제학 기법: 의인화, 디자인융복합연구, 17(3), 41-53  

<br>

## 2. Valence-Arousal

Valence-Arousal은 EEG, ECG, GSR 등의 생체신호를 바탕으로 하는 연구에 주로 활용되는 정서모델로, 모호할 수 있는 정서를 객관적인 지표로 표현할 수 있기에 선택하였다. Valence는 정서가에 해당하는 축으로 HRV(Heart Rate Variability)를 측정하여 얻어진다. 부정적인 정서에서 심박은 규칙적이고, 긍정적인 정서에서의 심박은 불규칙하게 나타나므로 높은 HRV는 긍정정서, 낮은 HRV는 부정정서를 암시한다. Arousal은 각성가로 GSR(Galvanic Skin Response)을 이용해 측정할 수 있다. 각성 정도가 높을수록 땀 분비가 많아지기 때문에 피부 전도도가 높아지게 되고, 이는 자연히 높은 각성 정도를 나타내는 지표가 된다. 사용된 lexicon과 corpus 역시 측정된 생체신호를 바탕으로 label되었기에 정서에 대한 객관적인 지표로 간주하였다.  

이 모델은 Arousal이 기존의 감성분석에서 가장 많이 활용되는 긍정-부정에 해당한다는 점에서 두 가지 이점을 취할 수 있다.  
1. 긍정-부정 연구에 활용되는 데이터를 재사용할 수 있다.
2. Binary 문제에서 하나의 차원을 늘리는 것으로 다양한 정서를 분류할 수 있다.

<br>

## 3. 데이터 수집 및 전처리

### 3.1 데이터
사용된 데이터는 아래의 두 가지이다.
- NRC-VAD(lexicon) : https://saifmohammad.com/WebPages/nrc-vad.html
- emobank(corpus) : https://github.com/JULIELab/EmoBank

NRC-VAD는 약 20k개의 단어에 Valence-Arousal-Dominance 점수가 0-1 사이로 측정된 lexicon으로 여기에서 Valence, Arousal 점수만 사용하였다. emobank는 약 10k개의 문장에 Valence-Arousal-Dominance 점수가 1-5 사이로 측정된 corpus로 여기에서 Valence, Arousal 점수만 사용하였다. stopword와 padding을 0으로 채우기 위해 모든 데이터의 label은 -1~1로 scaling하였다.

### 3.2 Embedding
embeding은 두 가지의 방법으로 시도하였다.
1. lexicon의 Valence-Arousal 점수를 그대로 embedding으로 사용  
: 표로 그리기
2. torchtext에 사전 학습되어있는 GloVe embedding을 사용

<br>

## 4. 모델 구축 및 학습
### 4.1 lexicon based trial
해당 문제는 regression으로 시도되었으며, 이는 상대적인 지표이기 때문에 기준으로 아래의 값에 대한 mse를 참고하였으며, 형용사를 제외한 모든 품사에 대해 `weight*0.1`을 적용했을 때의 mse가 평균 0.0295745652973668으로 가장 낮게 나타나 이후 학습에서도 같은 weight를 적용하였다.  

![image](https://user-images.githubusercontent.com/39390943/92320980-e522a100-f060-11ea-8635-edcfa1aebdc4.png)  

mse: 0.26928572972569476

#### 4.1.1 CNN
![image](https://user-images.githubusercontent.com/39390943/92321057-b48f3700-f061-11ea-8813-2f1ad5558245.png)

> **Test complete : avg. loss : 0.022307074116252004**

#### 4.1.2 Bi-LSTM
![image](https://user-images.githubusercontent.com/39390943/92321041-9a555900-f061-11ea-8b50-349c0ed57a5a.png)

> **Test complete : avg. loss : 0.08659133683582518**

#### 4.1.3 C-LSTM
![image](https://user-images.githubusercontent.com/39390943/92321067-d8527d00-f061-11ea-8c5e-4411c6f531f8.png)

> **Test complete : avg. loss : 0.025447068566789324**


### 4.2 GloVe embedding
lexicon을 그대로 embedding으로 사용한 모든 시도에서 mse는 기준삼은 점수보다 낮게 나타났지만, train loss는 모두 안정적으로 수렴하지 않았기에 제대로 학습이 이루어지지 않았음을 알 수 있다. 따라서 추가적인 시도로 torchtext에 사전학습되어있는 GloVe embedding을 사용했다. 사용된 embedding은 `glove.6B.300d`이다.

#### 4.2.1 CNN

#### 4.2.2 Bi-LSTM

#### 4.2.3 C-LSTM

<br>

## 5. 한계 및 추가 시도

최종적으로 가장 좋은 학습을 보인 것은 ____으로 Valence ____, Arousal ____, 평균 ____의 결과를 얻었다. 다만 이는 Valence-Arousal 점수를 나타낸 것으로 직관적으로 이해할 수 있는 정서를 의미하지는 않는다. 이에 추가 시도로 도전하고자 하는 것은 클러스터링을 이용해 구체적인 정서 카테고리를 분류하는 것이다. 자연어에서 multi-class classification 문제는 여전히 해결하기 어려운 문제로 남아있는데(SST-5 SOTA의 경우 정확도는 0.55), 해당 프로젝트는 비록 regression 문제라도 2개의 차원을 이용했기 때문에 차원에서의 이점을 기대하고 있다.
