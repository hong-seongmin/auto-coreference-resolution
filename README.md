# auto-coreference-resolution

## 개요

자동으로 상호참조해결을 할 수 있을지 생각해보는 레포입니다. 지금은 아이디어 단계입니다.

상호참조 해결(Corefence Resolution)은 문맥을 통해 각 대명사나 명사가 무엇을 지칭하는지 명확히 하는 과정입니다. 즉, 문장이나 문단 내에서 동일한 대상(사람, 장소, 사물 등)을 지칭하는 대명사나 명사를 찾아내고 이들 간의 연관성을 파악하는 작업입니다.

대명사를 찾아내기 위해 kiwi 라이브러리와 MASK를 채우기 위해 kakaobank/kf-deberta-base를 사용했습니다.

문장들을 분석하는 과정에서 대명사를 찾아 바로 바꾸면 띄어쓰기 등에 문제가 발생하기 때문에 먼저 위치와 길이를 기억하고 이후에 뒤의 단어들부터 바꾸는 방법을 사용했습니다.

```
# 대명사(NP) 위치와 길이 기억하기
pronoun_positions = []
for sent_idx, sent in enumerate(sentences):
    tokens = kiwi.tokenize(sent.text)
    for token in tokens:
        if token.tag == 'NP':
            pronoun_positions.append((sent_idx, token.start, token.end))
```

kf-deberta-base는 금융 도메인 특화 언어모델로 소개돼 있지만 개인적인 테스트 결과 가장 잘 예측을 하였습니다.

다만 최대 토큰 사이즈를 고려하여 맥락은 앞뒤 200정도로 잡았습니다.

```
# 앞쪽 문맥 추출
pre_context = ""
pre_length = 0
for i in range(sent_idx - 1, -1, -1):
    if pre_length + len(sentences[i].text) <= context_max:
        pre_context = sentences[i].text + " " + pre_context
        pre_length += len(sentences[i].text)
    else:
        break

# 뒤쪽 문맥 추출
post_context = ""
post_length = 0
for i in range(sent_idx + 1, len(sentences)):
    if post_length + len(sentences[i].text) <= context_max:
        post_context += " " + sentences[i].text
        post_length += len(sentences[i].text)
    else:
        break
```

Fill-Mask작업에서 첫번째 단어가 명사가 아닌 경우가 있습니다. 따라서 이 때 가장 높은 순위의 명사를 찾아 바꿔주면 정확도가 조금 더 올라갑니다.

```
# 모델 입력 및 예측
inputs = tokenizer(context_sentence, return_tensors="pt").to(device)
outputs = model(**inputs)
logits = outputs.logits
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()

# 예측된 상위 5개 후보 토큰 추출
top_k_tokens = torch.topk(torch.nn.functional.softmax(logits[0, mask_token_index], dim=-1), 5).indices
predicted_tokens = [tokenizer.decode([idx]) for idx in top_k_tokens]

# 명사만 선택
selected_token = predicted_tokens[0]  # 기본값: 첫 번째 후보
for token in predicted_tokens:
    analyzed = kiwi.tokenize(token)
    if analyzed[0].tag in ['NNG', 'NNP', 'NNB']:  # 명사 태그들
        selected_token = token
        break
```

## 결과

동양철학은 고대 중국과 인도에서 시작된 철학적 사상 체계이다. 동양철학의 중심
에는 자연과 인간의 조화, 도덕적 원칙, 그리고 인간의 내적 수양이 있다. 공자는
"인간은 도덕적 원칙을 따르는 것이 중요하다"고 가르쳤다. 공자는 제자들에게 "공자가 가르친 도는 간단하지만, 실천하기는 어렵다"고 말하곤 했다. 또 다른 철
학자인 노자는 "자연과 하나가 되는 것이 진정한 행복의 길이다"라고 주장했다.  
노자의 철학은 무위자연, 즉 인위적인 노력이 아닌 자연스러운 상태에서의 삶을  
강조했다. 노자의 철학은 이후 많은 철학자들에게 영향을 미쳤다.
Context: 또 다른 철학자인 노자는 "자연과 하나가 되는 것이 진정한 행복의 길이
다"라고 주장했다. 그의 철학은 무위자연, 즉 인위적인 노력이 아닌 자연스러운  
상태에서의 삶을 강조했다. 그의 철학은 이후 많은 [MASK]들에게 영향을 미쳤다.
Original: 이
Predicted: 철학자
Context: 또 다른 철학자인 노자는 "자연과 하나가 되는 것이 진정한 행복의 길이
다"라고 주장했다. 그의 철학은 무위자연, 즉 인위적인 노력이 아닌 자연스러운  
상태에서의 삶을 강조했다. [MASK]의 철학은 이후 많은 이들에게 영향을 미쳤다.
Original: 그
Predicted: 노자
Context: 그는 제자들에게 "내가 가르친 도는 간단하지만, 실천하기는 어렵다"고
말하곤 했다. 또 다른 철학자인 노자는 "자연과 하나가 되는 것이 진정한 행복의
길이다"라고 주장했다. [MASK]의 철학은 무위자연, 즉 인위적인 노력이 아닌 자연
스러운 상태에서의 삶을 강조했다. 그의 철학은 이후 많은 이들에게 영향을 미쳤
다.
Original: 그
Predicted: 노자
Context: 공자는 "인간은 도덕적 원칙을 따르는 것이 중요하다"고 가르쳤다. 그는
제자들에게 "[MASK]가 가르친 도는 간단하지만, 실천하기는 어렵다"고 말하곤 했
다. 또 다른 철학자인 노자는 "자연과 하나가 되는 것이 진정한 행복의 길이다"라
고 주장했다. 그의 철학은 무위자연, 즉 인위적인 노력이 아닌 자연스러운 상태에
서의 삶을 강조했다.
Original: 내
Predicted: 공자
Context: 공자는 "인간은 도덕적 원칙을 따르는 것이 중요하다"고 가르쳤다. [MASK]는 제자들에게 "내가 가르친 도는 간단하지만, 실천하기는 어렵다"고 말하곤 했
다. 또 다른 철학자인 노자는 "자연과 하나가 되는 것이 진정한 행복의 길이다"라
고 주장했다. 그의 철학은 무위자연, 즉 인위적인 노력이 아닌 자연스러운 상태에
서의 삶을 강조했다.
Original: 그
Predicted: 공자

## 결론

아주 만족스러운 수준까지는 아니지만 쉬운 문장들에 대해서는 시도해볼만한 작업인 것 같습니다.

더 나은 방법이나 모델이 있다면 알려주시면 감사하겠습니다.

e-mail: hong112424@naver.com
작성자: 솔론(solon)
