---
title: Rust로 감성분석 AI 구현하기 (LSTM, GRU)
date: 2024-05-13
tags:
  - AI
  - LSTM
  - GRU
  - Rust
  - Candle
  - Huggingface
  - Transformers
  - PyTorch
authors:
  - 이재희
slug: sentiment-ai-implementation-with-rust
---
# 개요
파이썬은 내가 제일 많이 사용하는 언어 중 하나이다. 또한 파이썬은 PyTorch, Tensorflow, Keras 등 굉장히 편리하고 사용하기 쉬운 라이브러리가 많다. 하지만 대부분의 파이썬 머신러닝 라이브러리의 중심 부분은 파이썬으로 구현되지 않고 C, C++ 등의 저수준 언어로 구현되어있다. 파이썬은 인공신경망 등의 알고리즘을 실행시키기에는 너무나도 느리기 때문이다. 

PyTorch는 인공지능 개발에 필수적인 라이브러리라고 할 수 있다. 파이토치에서 제공하는 수백, 혹은 수천 개의 머신러닝을 위한 API는 코드 몇 줄로 모델을 만들거나, 학습시키고 추론하기 위한 다양한 함수들을 제공한다. 실제로 구글 트렌드의 관심도 변화를 보면 2022년 6월을 기준으로 파이토치가 텐서플로우를 앞서는 모습을 보인다. 

그런데, 파이토치나 텐서플로우는 지원하는 기능들이 워낙 많다보니 cpu 환경에서 파이토치를 설치하려면 최소 600MB의 데이터를 다운로드 해야 한다. 그래서 나는 모델 학습은 고사양 컴퓨터에서, 모델 추론은 보다 저사양의 컴퓨터에서 하는 방식이 좀 더 효율적이라고 생각했다. 일반적으로 딥러닝 모델은 추론보다 학습에 더 많은 컴퓨팅 리소스를 사용하기 때문이다. 

HuggingFace는 [Tokenizers](https://github.com/huggingface/tokenizers) 라이브러리를 러스트로 구현하였고, 작년부터 [Candle](https://github.com/huggingface/candle) 프로젝트를 개발하고 있는데, 이 Candle은 러스트로 만든 머신러닝 프레임워크이다. 그리고 함수 이름과 인덱싱 등을 파이토치와 비슷하게 만들어놔서, 기존 파이토치로 구현한 모델을 러스트로 다시 구현하기에 어려움이 없도록 설계했다는 점이 가장 큰 장점인 것 같다. 파이토치와 비교했을 때 매우 가볍고, CPU 환경에서 작은 사이즈의 단일 실행 파일로 컴파일되어 상당히 마음에 들었다. 그래서 Candle을 사용하여 네이버 영화 리뷰([NSMC](https://github.com/e9t/nsmc)) 분류 모델을 만들어 보려고 한다.

# 학습 환경
- Ubuntu 22.04
- Python 3.10.14
- PyTorch 2.2.0+cu121
- Pandas
- Tokenizers
- Numpy
- tqdm
- safetensors

# 추론 환경
- Windows 11
- Rust 1.78.0
	- candle 0.5.1
	- tokenizers 0.19.1
	- anyhow 1.0.83


# 데이터셋 준비
네이버 영화 리뷰 데이터셋(NSMC, Naver Sentiment Movie Corpus)은 [여기](https://github.com/e9t/nsmc)에서 다운로드 받을 수 있다. 필요한 파일은 `ratings_train.txt`이다.

# 모델 구현 및 학습
## Python

### 토크나이저 학습
제일 먼저 토크나이저를 학습시켜야 한다.
```python
from tokenizers import ByteLevelBPETokenizer
```
나는 한글 데이터를 학습시킬 것이므로 BPE 토크나이저가 아무래도 더 적합하다고 생각했다.

```python
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(['ratings_train.txt'], vocab_size=10000)
```
토크나이저의 vocab 크기를 1만개로로 정의하고 "데이터셋 준비" 에서 다운받은 `ratings_train.txt` 파일을 인자로 넘기면 간단하게 학습이 가능하다. 학습하는 데 대략 30초 정도 소요된다. 

```python
tokenizer.encode("재미있는 영화군요").ids
```
잘 학습이 됐는지 테스트하기 위해 아무 문장이나 인코딩 해본다.
출력은 다음과 같다.
```python
[7457, 20609]
```

```python
tokenizer.save("tokenizer.json")
```
학습이 완료된 토크나이저를 JSON 형식으로 저장한다.

### 모델링
```python
import torch
import torch.nn as nn
```

다음으로 모델링에 필요한 라이브러리를 불러온다. 나는 LSTM, GRU와 같은 인공신경망 알고리즘이 필요하기 때문에 torch.nn 패키지를 nn만 쳐도 쉽게 사용할 수 있도록 하였다.

```python
hidden_dim = 128
output_dim = 128
num_labels = 2
class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(10000, hidden_dim)
        self.gru = nn.GRU(hidden_dim, output_dim, batch_first=True)
        self.ln1 = nn.Linear(outptu_dim, num_labels)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        last_hidden = hidden.squeeze(0)
        logits = self.ln1(last_hidden)
        return logits
```
모델 구현 부분은 간단하다. `TextClassifier` 클래스가 nn.Module 클래스를 상속하도록 하고, 레이어들을 정의하면 된다. 제일 먼저 `embedding` 레이어는 아까 학습한 토크나이저에 1만개의 vocab이 존재하므로 입력 부분은 토크나이저와 같이 10000을 인자로 넘겼다. 임베딩을 거쳐서 GRU(혹은 LSTM) 레이어에 값이 전달되고, 최종적으로 1과 0(긍정 혹은 부정)을 출력해야 하기 때문에 Linear 레이어의 출력을 2로 설정했다.
GRU의 batch_first 부분은 `(seq, batch, feature)`로 된 데이터를 `(batch, seq, feature)`처럼 배치 사이즈가 맨 앞으로 오게 하라는 뜻이다.

그리고 forward 함수에는 `__init__`에서 정의했던 레이어들에 데이터를 전달하는 코드를 작성한다.

### 모델 학습
```python
import torch
import pandas as pd
import torch.nn as nn
from tokenizers import Tokenizer
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from safetensors.torch import save_file
```
필요한 라이브러리들을 불러온다.

```python
df = pd.read_csv('ratings_train.txt', sep='\t')
```
이전에 다운받은 텍스트 파일은 사실 raw 형식이 아니라 tsv 형식이었다.
`df.head()`를 실행하면 다음과 같이 영화 리뷰 앞부분을 미리 볼 수 있다.

| id       | document                           | label |
| -------- | ---------------------------------- | ----- |
| 9976970  | 아 더빙.. 진짜 짜증나네요 목소리                | 0     |
| 3819312  | 흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나  | 1     |
| 10265843 | 너무재밓었다그래서보는것을추천한다                  | 0     |
| 9045019  | 교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정      | 0     |
| 6483659  | 사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서... | 1     |

```python
tokenizer = Tokenizer.from_file('tokenizer.json')
```
아까 저장된 tokenizer.json 파일을 불러온다.

```python
dataset = [tokenizer.encode(x).ids for x, y in zip(list(df['document']), list(df['label'])) if str(x) != 'nan' and str(y) != 'nan']

labels = [y for x, y in zip(list(data['document']), list(data['label'])) if str(x) != 'nan' and str(y) != 'nan']

```
귀찮아서 한줄로 짰더니 이 꼬라지가 나버렸다. 대충 데이터셋의 document 부분이랑 label 부분을 가져오면서 둘 다 값이 존재하면 리스트에 추가하는 코드라고 이해하면 된다.

`dataset[0]`을 입력하면 `[333, 2547, 262, 617, 4518, 480, 4492]`와 같이 정수로 인코딩된 문장이 나타나고, `labels[0]`을 입력하면 1 또는 0이 출력된다.

```python
max([len(x) for x in dataset])
```
모든 문장의 최대 길이를 출력해보면 117이 나온다. 즉 정수로 인코딩 된 데이터셋은 길이가 117을 초과하지 않는다는 뜻이다. 

```python
def pad_sequences(sentences, max_len):
    features = np.zeros((len(sentences), max_len), dtype=int)
    for i, sentence in enumerate(sentences):
        if len(sentence) != 0:
            features[i, :len(sentence)] = np.array(sentence)[:max_len]
    return features

padded = pad_sequences(dataset, 128)
```

넉넉하게 128로 패딩하였다. 패딩 후 `train_dataset`의 맨 첫 번째 항목은 아래와 비슷하게 나올 것이다.
```python
array([ 333, 2547,  262,  617, 4518,  480, 4492,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0])
```


```python
train_dataset = TensorDataset(torch.tensor(padded), torch.tensor(labels))
train_loader = DataLoader(train_dataset, batch_size=32)
```
TensorDataset을 사용해 정수 리스트로 된 데이터셋을 파이토치에서 사용하는 tensor 형식으로 변환한다.
그 다음 병렬 학습을 위해 DataLoader에 배치 사이즈를 인자로 넣어 몇 개의 데이터를 병렬로 처리할 건지 설정한다.

```python
model = TextClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 6
model.to('cuda' if torch.cuda.is_available() else 'cpu')
```
"모델링"에서 구현한 `TextClassifier` 모델을 생성하고, loss 함수와 optimizer, 반복 학습 횟수(num_epochs)를 정한다. 만약 cuda 지원 그래픽카드가 있다면, `model.to`를 사용하여 GPU에 모델을 올릴 수 있다.

```python
for epoch in range(num_epochs):
    train_loss = 0
    model.train()
    for i, (x, y) in tqdm(enumerate(train_loader)):
        x, y = x.to('cuda'), y.to('cuda')
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if i % 50 == 0 and i != 0:
            tqdm.write(f"Loss: {train_loss / i}\r", end='')
    print(f"Epoch Loss: {train_loss / len(train_loader)}")
```

드디어 학습을 진행한다. 50스텝, 1에폭마다 loss를 출력하게 구현하였다.
그러면 다음과 같이 결과가 출력되는데, 마음이 편안해진다.
```
Epoch Loss: 0.44622183628793516
Epoch Loss: 0.3024691305053341
Epoch Loss: 0.23970229042306188
Epoch Loss: 0.18055053850287833
Epoch Loss: 0.13453298333506641
Epoch Loss: 0.10856294956053496
```

마지막으로 학습한 모델을 safetensors 형식으로 내보내면 끝이다.
```python
save_file(model.state_dict(), "nsmc.safetensors")
```

## Rust

이미 학습을 진행했고, 모델 파일까지 얻었으므로 Rust 코드에서는 모델만 구현하면 된다.
`cargo new classification`과 같이 프로젝트를 새로 만들고, `Cargo.toml` 파일의 `[dependencies]` 항목에 다음과 같이 라이브러리를 추가한다.

```toml
[dependencies]
anyhow = "1.0.83"
candle-core = { git = "https://github.com/huggingface/candle.git", version = "0.5.1" }
candle-nn = { git = "https://github.com/huggingface/candle.git", version = "0.5.1" }
tokenizers = "0.19.1"
```
버전은 달라질 수 있으므로 공식 문서를 참조하여 최신 버전 사용을 권장한다.

`Cargo.toml`에 다음 코드를 추가한다.
```toml
[features]
cuda = ["candle-core/cuda", "candle-nn/cuda"]
```
만약 시스템이 CUDA를 지원한다면, feature 플래그를 통해 candle의 cuda를 활성화하는 부분이다.

프로젝트 내의 `main.rs` 파일의 내용을 전부 지우고 처음부터 시작하는 것이 편하다.

```rust
use std::io::Write;

use anyhow::Error as E;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn as nn;
use nn::{Module, VarBuilder, RNN};
use tokenizers::Tokenizer;
```
필요한 crate를 추가한다.

```rust
#[derive(Clone)]
struct TextCLassifier {
    embedding: nn::Embedding,
    gru: nn::GRU,
    ln1: nn::Linear,
    device: Device,
}
```
파이썬과 같이 TextClassifier 모델 구조를 정의한다.

```rust
impl TextCLassifier {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let embedding = nn::embedding(10000, 128, vs.pp("embedding"))?;
        let gru = nn::gru(128, 128, Default::default(), vs.pp("gru"))?;
        let ln1 = nn::linear(128, 2, vs.pp("ln1"))?;
        let device = Device::cuda_if_available(0)?;
        return Ok(Self {
            embedding,
            ln1,
            gru,
            device,
        });
    }
    pub fn forward(self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.embedding.forward(xs)?;
        let mut gru_states = vec![self.gru.zero_state(1)?];
        for x in &xs.squeeze(0)?.to_vec2::<f32>()? {
            let state = self.gru.step(
                &Tensor::from_vec(x.clone(), (1, x.len()), &self.device)?,
                &gru_states.last().unwrap(),
            )?;
            gru_states.push(state);
        }
        let xs = gru_states.last().unwrap().h();
        let xs = self.ln1.forward(&xs)?;
        Ok(xs)
    }
}
```
파이썬의 `__init__`과 같이 `new` 함수를 정의하고, 파이썬에서 구현한 모델에 맞춰 레이어 등을 똑같이 구현해야 한다. 
`forward` 함수의 내용은 비슷하다.
추가로 터미널 내에서 입력 & 추론하려면 다음 코드를 `impl TextClassifier` 블록 안에 넣으면 된다.
```rust
pub fn interaction(&mut self, tokenizer: Tokenizer, device: &Device) -> Result<()> {
	loop {
		let mut line = String::new();
		print!("영화 리뷰를 입력하세요: ");
		std::io::stdout().flush()?;
		std::io::stdin().read_line(&mut line)?;
		if line == "q" || line == "exit" || line == "quit" {
			break;
		}
		let encoded = tokenizer.encode(line, false).map_err(E::msg)?;
		let data = Tensor::new(vec![encoded.get_ids()], device)?;
		let result = self.clone().forward(&data)?;
		let result = result.argmax(1)?.to_vec1::<u32>()?;
		if result[0] == 0 {
			println!("부정적인 리뷰");
		} else if result[0] == 1 {
			println!("긍정적인 리뷰");
		} else {
			println!("알 수 없음");
		}
	}
	Ok(())
}
```



```rust
fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let vs = unsafe {
        nn::VarBuilder::from_mmaped_safetensors(&["nsmc.safetensors"], DType::F32, &device)?
    };
    let mut model = TextCLassifier::new(vs)?;
    let tokenizer = Tokenizer::from_file("tokenizer.json").map_err(anyhow::Error::msg)?;
    model.interaction(tokenizer, &device)?;\
    Ok(())
}
```
메인 함수에서 safetensors 포맷으로 저장된 모델 파일과 토크나이저를 가져온 후, 생성한 모델의 interaction 함수를 사용하면 터미널과 상호작용 할 수 있다. 

# 끝

**실제 사용 예시**
```raw
영화 리뷰를 입력하세요: 와 진짜 재밌는 영화임미다!!
긍정적인 리뷰
영화 리뷰를 입력하세요: 이딴걸 영화라고 처 만들었냐?
부정적인 리뷰
영화 리뷰를 입력하세요: 감독 나와라 ㅋㅋ
부정적인 리뷰
영화 리뷰를 입력하세요: 이 정도면 괜찮은 영화 아닌가?
긍정적인 리뷰
영화 리뷰를 입력하세요: 감동적이네료 ㅠㅜㅜ
긍정적인 리뷰
영화 리뷰를 입력하세요: 볼 가치도 없는듯 ㅋ
부정적인 리뷰
```

무려 5MB 정도의 작은 모델인데, 직접 영화 리뷰를 써보니 정확도가 90% 이상 나온다고 느꼈다. 거대 언어 모델에 비하면 리뷰를 긍정 또는 부정으로 분류하는 작업밖에 할 수 없어 초라해 보이지만 속도가 중요한 실제 서비스, 또는 임베디드 분야에서 유용할 것 같다는 생각이 들었다. 
사실 성공할 줄은 몰랐는데, 직접 모델을 구현하며 여러 시행착오를 겪었던 점에서 의미가 있는 것 같다.

