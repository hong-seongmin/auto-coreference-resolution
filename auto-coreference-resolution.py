from kiwipiepy import Kiwi
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def resolution_coreference(texts, context_max=200, is_gui=False):
    kiwi = Kiwi()

    # 문장을 형태소 분석 및 문장 단위로 쪼개기
    sentences = kiwi.split_into_sents(texts)

    # 대명사(NP) 위치와 길이 기억하기
    pronoun_positions = []
    for sent_idx, sent in enumerate(sentences):
        tokens = kiwi.tokenize(sent.text)
        for token in tokens:
            if token.tag == 'NP':
                pronoun_positions.append((sent_idx, token.start, token.end))

    tokenizer = AutoTokenizer.from_pretrained("kakaobank/kf-deberta-base")
    model = AutoModelForMaskedLM.from_pretrained("kakaobank/kf-deberta-base").to(device)

    resolved_text = texts
    resolving = []

    # 각 [MASK] 위치에 대해 앞뒤로 최대 context_max 글자씩 문맥 추출 및 처리
    for sent_idx, token_start, token_end in tqdm(reversed(pronoun_positions), desc="Processing [MASK] tokens"):
        # [MASK]로 변환된 문장 생성
        masked_sentence = sentences[sent_idx].text[:token_start] + '[MASK]' + sentences[sent_idx].text[token_end:]

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

        context_sentence = pre_context.strip() + ' ' + masked_sentence + ' ' + post_context.strip()

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

        # 원문에 예측된 대명사 대입
        original_start = sentences[sent_idx].start + token_start
        original_end = sentences[sent_idx].start + token_end
        resolved_text = resolved_text[:original_start] + selected_token + resolved_text[original_end:]
        resolving.append([context_sentence, texts[original_start:original_end], selected_token])

    return resolved_text, resolving

# 예제 텍스트
text_raw_kor = """동양철학은 고대 중국과 인도에서 시작된 철학적 사상 체계이다. 동양철학의 중심에는 자연과 인간의 조화, 도덕적 원칙, 그리고 인간의 내적 수양이 있다. 공자는 "인간은 도덕적 원칙을 따르는 것이 중요하다"고 가르쳤다. 그는 제자들에게 "내가 가르친 도는 간단하지만, 실천하기는 어렵다"고 말하곤 했다. 또 다른 철학자인 노자는 "자연과 하나가 되는 것이 진정한 행복의 길이다"라고 주장했다. 그의 철학은 무위자연, 즉 인위적인 노력이 아닌 자연스러운 상태에서의 삶을 강조했다. 그의 철학은 이후 많은 이들에게 영향을 미쳤다."""
resolved_text, resolving = resolution_coreference(text_raw_kor, context_max=100)

print(resolved_text)
for context_sentence, original, predicted in resolving:
    print(f"Context: {context_sentence}")
    print(f"Original: {original}")
    print(f"Predicted: {predicted}")
