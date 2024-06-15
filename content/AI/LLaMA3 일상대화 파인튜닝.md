---
title: Efficient LLaMA3 Fine-Tuning with Unsloth
date: 2024-05-10
tags:
  - LLaMA
  - LLM
  - AI
  - Fine-Tuning
  - Unsloth
  - Conversational
authors:
  - 이재희
slug: unsloth-llama3-fine-tuning
---
# 개요
글을 작성하는 현 시점 기준으로 약 2주 전 [라마3](https://llama.meta.com/llama3/)가 발표되었다. 성능이 어떨지 궁금했기 때문에 이번 기회에 라마3을 파인튜닝하자고 결심했다. 허깅페이스에 누군가 한국어 데이터셋에 맞춰 학습시켜놓은 모델이 있었기 때문에 파인튜닝 시 기본 영어 모델보다 한국어 성능이 나을 것이라고 판단하여 [이 모델](https://huggingface.co/beomi/Llama-3-Open-Ko-8B)을 베이스로 사용하기로 했다. 

# 학습 전 준비물
- **개발 환경 (GPU 클라우드)**
	- RTX 4090 X 1
	- Ubuntu 22.04
	- PyTorch 2.2.0
	- CUDA 12.1
- **충분한 자본금(4090 기준 시간 당 약 0.3~0.4달러)**
- **초연한 자세**
	- 어떠한 오류가 터져도 화내지 않는 강건한 정신
- **채팅 데이터셋**
	- [AI Hub](https://aihub.or.kr)에서 가져온 **주제별 텍스트 일상 대화 데이터**
	- 데이터 전처리
- **라이브러리(pypi)**
	- [unsloth](https://github.com/unslothai/unsloth)
	- accelerate
	- transformers
	- trl
	- datasets
	- peft
	- etc.

# 채팅 데이터셋 전처리
JSON으로 구성된 채팅 데이터셋은 대충 전처리하면 다음과 같은 텍스트 형태가 된다.
```
user: 점심 메뉴 정하신 분
bot: 누룽지 끓여 먹을까 고민
user: 나 아침 든든하게 먹으니 힘 난다
user: 아 나 점심 뭐 먹지
bot: 점심은 사모님이 주심 후후
user: 나 점심 내장국밥 픽
user: **가 오늘 만들어야 함?
bot: 누룽지는 간식으로 먹어야징
user: 돼지국밥으로 바꿀까
user: 누룽지 맛있겠어... 나두
bot: 여기 먹을 거 개많음 ㄷㄷ
user: 아니면 두루치기 먹을가
user: 국밥은 언제나 맛있음
bot: 떡볶이 이런 거 먹을 수 있음 키키
user: 먹는 게 제이 좋아!
user: 국밥 1그릇 배달 안 해주겠지 ㅠ
bot: 수제비랑 밀키트 다 나왕
user: 떡볶이 파는 데가 없음 여기
<|endoftext|>
```
여기서 문제가 하나 발생한다. 우리는 로봇과 1대1로 대화하기를 원하는데, 이 데이터셋은 화자가 최대 3명이기 때문에 1번 화자는 `user`, 2번 화자는 `bot`, 3번 화자는 다시 `user`로 치환해버려서 user는 이중인격자가 되는 것을 확인할 수 있다. 이 부분은 학습이 완료된 후 알아채서 다음에는 `user1, user2, user3, ...`이런 식으로 시도해보려고 한다.
또한 이름도 개인정보 보호 차원에서 별 모양 기호(asterisk, \*)로 뜨는데, 적당하게 \<name>으로 바꿨으면 더 좋았을 것 같다.
마지막으로 오타가 상당히 많은데, 오타 좀 내면 더욱 사람과 비슷해지지 않을까 하는 생각이 들어 오타 교정은 수행하지 않았다. 사실 몇만 줄의 텍스트 파일을 맞춤법 검사기를 돌려 일일이 수정하는 것도 미친 짓이라고 생각했다.
# 학습 코드
```python
import torch
print(torch.__version__)
from tqdm.auto import tqdm
from unsloth import FastLanguageModel
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments
```
먼저 필요한 모듈을 import한다. 여기서 unsloth 모듈은 뭐하는 놈일까?

## Unsloth는 무엇인가
unsloth는 여러 언어모델들을 파인튜닝 시 2배~5배 빠르게, 메모리 사용량을 80%까지 줄일 수 있다고 주장하는 라이브러리다. 실제로 vram 사용량을 꽤 줄였던 것으로 기억한다.

| 1 A100 40GB | 🤗Hugging Face | Flash Attention | 🦥Unsloth Open Source | 🦥[Unsloth Pro](https://unsloth.ai/pricing) |
| ----------- | -------------- | --------------- | --------------------- | ------------------------------------------- |
| Alpaca      | 1x             | 1.04x           | 1.98x                 | **15.64x**                                  |
| LAION Chip2 | 1x             | 0.92x           | 1.61x                 | **20.73x**                                  |
| OASST       | 1x             | 1.19x           | 2.17x                 | **14.83x**                                  |
| Slim Orca   | 1x             | 1.18x           | 2.22x                 | **14.82x**                                  |

위 표는 unsloth의 깃허브 readme에서 가져온 벤치마크 결과이다. 최대 2배 빠르다는 건 이해할 수 있지만, 아무리 봐도 14~15배 빨라진다는 unsloth pro는 그짓말 같긴 하다. 

아무튼 unsloth는 파이토치 버전, 쿠다 버전, GPU Compute Capability에 따라 설치 명령어가 약간씩 다르니 꼭 [README](https://github.com/unslothai/unsloth)를 참고하여 설치하자.

## 모델 & 토크나이저 다운로드
```python
model, tokenizer = FastLanguageModel.from_pretrained('beomi/Llama-3-Open-Ko-8B', max_seq_length=1024, dtype=None, load_in_4bit=True)
```
`unsloth`에서 제공하는 `FastLanguageModel`을 사용하여 `transformers` 라이브러리와 비슷한 방식으로 모델을 다운로드 받을 수 있다.
그리고 `load_in_4bit` 인자를 사용하여 모델을 4비트 양자화한 상태로 불러온다. 양자화를 하지 않으면 나중에 학습할 때 VRAM 점유율이 4090의 24GB를 초과하여 OOM(CUDA Out Of Memory) 오류가 터진다. 다만 양자화로 인해 모델의 성능 저하도 있는 것 같다. `max_seq_length`는 입력 토큰의 최대 길이이다. 1024 토큰이 좀 작을 수도 있지만, OOM이 무서워서 일단 1024로 했다. 나중에 RoPE Scaling이라는 기술을 활용하면  모델의 context size를 파인튜닝 없이 늘릴 수 있다고 한다.

## PEFT 모델로 변환
```python
model = FastLanguageModel.get_peft_model(
    model, 
    r=16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj",],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    max_seq_length=1024,
    use_rslora=False,
    loftq_config=None,
)
```
PEFT(Parameter Efficient Fine-Tuning)는 Huggingface 커뮤니티에서 개발 중인 라이브러리다. PEFT는 모델의 전체 파라미터를 학습하는 대신, LoRA와 같은 기술을 사용하여 필요한 작은 수의 파라미터만 학습해서 컴퓨팅 비용을 줄일 수 있는 모듈이다. 기존 학습 코드를 조금만 수정하면 효율적으로 학습이 가능하다는 점이 장점이다. 
Unsloth는 PEFT 모듈 + $\alpha$를 활용해서 속도와 메모리 사용량을 줄이는 방식이기 때문에 두 라이브러리는 다르다.

Unsloth에서 제공하는 `FastLanguageModel`에서 모델을 PEFT 모델로 변환할 수 있다. 내부적으로 LoRA를 사용하는 듯하다. LoRA 기술에 대해서는 기본적인 동작 원리만 이해하고 자세히는 알지 못해서 나중에 따로 공부해보고 싶다.

## 데이터셋 불러오기
```python
dataset = load_dataset('json', data_files='chat.json', split='train')
```
huggingface `datasets` 모듈으로 아까 전처리한 텍스트 파일을 대화 주제 단위로 자르고, 다음과 같은 JSON 형식으로 변환해서 `load_dataset` 함수로 불러왔다.
```json
[
	{
		"text": "user: 점심 메뉴 정하신 분 bot: 누룽지 끓여 먹을까 고민...[이하생략]>"
	}
]
```

## Trainer 구현
```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field='text',
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        bf16=True,
        logging_steps=10,
        output_dir='results',
        optim='adamw_8bit',
    ),
)
```
trl에서 제공하는 SFTTrainer(Supervised Fine-Tuning Trainer)를 사용하면 쉽게 파인튜닝할 수 있다. 여기서 중요한 코드가 `TrainingArguments` 부분이라고 생각하는데, 학습에 필요한 learning rate 등을 TrainingArguments에 인자로 넘기는 부분이기 때문이다. `per_device_train_batch_size` 대신 `auto_find_batch_size=True`를 쓰면 자동으로 최적의 batch size를 찾아준다. 사실 `TrainingArguments`는 unsloth의 readme를 귀찮아서 대충 그냥 갖다 쓴거라 뭔지는 나도 잘 모른다.

## 학습
이제 학습은 굉장히 간단해진다. 
```python
trainer.train()
```
끝.

Unsloth가 편리한게 다음과 같이 코드를 입력하면 자동으로 llama.cpp에서 돌아가는 GGUF 파일 형식으로 변환하고 양자화까지 해준다는 점이다.

```python
model.save_pretrained_gguf("dir", tokenizer, quantization_method = "q4_k_m")
model.save_pretrained_gguf("dir", tokenizer, quantization_method = "q8_0")
model.save_pretrained_gguf("dir", tokenizer, quantization_method = "f16")
```

# 학습 결과
이제 챗봇과 대화를 해보자.

> **대화 기록**
> ```raw
> user: 안녕
> bot: 안녕
> user: 오늘 날씨 좋다 ㅎㅎ
> bot: 오늘 날씨가 너무 좋아 키키
> user: 밖에 나가자
> bot: 그러게 완전 가을이야~
> user: 여름인데?
> bot: 여름이 아니지 않나?
> user: 여름맞는데?
> bot: 아직 여름인 거 같은데?
> user: 지금 내 말이 틀렸다고 주장하는 거니?
> bot: 그렇다고 할 수는 없잖아~ 나도 좋아 키키
> ```

내가 파인튜닝을 너무 대충 해서 그런지는 모르겠는데, 대화를 하면 할 수록 배가 산으로 가는 현상을 볼 수 있다. 
아니면 약 4MB 용량의 적은 데이터셋으로 학습을 진행해서 성능이 구린 이유도 있는 것 같다.

# 마치며
대략 2년 전부터 진짜 사람같은 챗봇을 만들고 싶어서 여러가지 시도를 해보고 있다. ChatGPT 등 성능이 좋은 모델의 api를 사용하면 되지 않느냐고 생각할 수 있지만, API를 사용하는 방식은 내 성미에 맞지 않기도 하고 gpt 특유의 어색한 말투가 너무 부자연스러워서 대화에 몰입할 수 없었다. 
재미있는 건 1년 전에 EleutherAI의 `polyglot-ko`라는 pretrained 모델을 사용해서 파인튜닝한 결과물이 지금보다 좋았다고 느꼈다는 사실이다. 아무래도 polyglot-ko는 완전히 처음부터 한국어 데이터로 pretraining한 반면, 라마와 같은 모델은 영어의 비중이 높고, 한국어가 거의 없는 데이터셋으로 학습 후 누군가 한국어로 다시 파인튜닝했기 때문에 한국어 능력이 부족하지 않았나 추측해본다. 
라마와 같은 pretrained 모델 하나를 만드려면 엄청난 비용과 시간이 든다. 그럼에도 불구하고 2022년에 공개된 polyglot-ko 이후 처음부터 한국어로 학습한 모델이 전무하다고 할 수 있을 정도로 현재 한국어 LLM 생태계는 기존 영어 모델을 파인튜닝하고, 다시 파인튜닝하는 부분에 초점이 맞춰져있는 것 같다. 그래서 나는 네이버 등의 회사가 비교적 작은 모델인 한국어 sLLM을 공개했으면 좋겠다는 바람이 있다.

> 개인적인 의견이므로 부정확할 수 있으니 참고하지 마시오.

