# Fine-tuning 가이드

## 개요

학습 데이터(`training_data.jsonl`)로 Qwen2.5-Coder-7B 모델을 파인튜닝하는 방법입니다.

---

## 방법 1: Google Colab (권장)

### 필요 사항
- Google Colab Pro ($10/월) 또는 무료 버전
- GPU: T4 (무료) 또는 A100/V100 (Pro)

### 실행 단계

1. **Colab 노트북 업로드**
   - [finetune_html_ui_colab.ipynb](./finetune_html_ui_colab.ipynb)를 Colab에 업로드

2. **GPU 런타임 설정**
   - `런타임` → `런타임 유형 변경` → `GPU` 선택

3. **학습 데이터 업로드**
   ```
   data/training/training_data_20260513_162533.jsonl
   ```
   - 파일명을 `training_data.jsonl`로 변경하여 업로드

4. **셀 순서대로 실행**
   - 학습 시간: T4 기준 약 2-4시간

5. **GGUF 파일 다운로드**
   - 생성된 `qwen-html-ui-gguf.zip` 다운로드

---

## 방법 2: 로컬 GPU (RTX 3090/4090)

**요구 사항**: VRAM 24GB 이상

```bash
# 가상환경 활성화
.\.venv\Scripts\activate

# 필수 패키지 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft bitsandbytes trl

# 학습 실행
python training/train_qlora_colab.py
```

---

## 방법 3: 클라우드 GPU 서비스

### Lambda Labs / RunPod / Vast.ai

```bash
# 1. A100 40GB 인스턴스 생성

# 2. 환경 설정
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# 3. 데이터 업로드
scp data/training/training_data.jsonl user@server:/workspace/

# 4. 학습 실행
python training/train_unsloth_colab.py
```

---

## 학습 후 로컬 사용

### 1. GGUF 파일 준비

Colab에서 다운로드한 `qwen-html-ui-gguf.zip` 압축 해제:

```bash
unzip qwen-html-ui-gguf.zip -d models/
```

### 2. Ollama에 모델 등록

```bash
cd models/qwen-html-ui-gguf
ollama create html-ui-agent -f Modelfile
```

### 3. 테스트

```bash
ollama run html-ui-agent "로그인 페이지 만들어줘"
```

---

## 하이퍼파라미터 조정

### 학습 데이터가 부족할 때 (< 5,000개)
```python
NUM_EPOCHS = 5          # 에포크 증가
LEARNING_RATE = 1e-4    # 학습률 낮춤
```

### 데이터가 충분할 때 (> 10,000개)
```python
NUM_EPOCHS = 2          # 에포크 감소
BATCH_SIZE = 8          # 배치 증가 (A100 기준)
```

### 메모리 부족 시
```python
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
MAX_SEQ_LENGTH = 1024   # 시퀀스 길이 축소
```

---

## 예상 학습 시간

| GPU | 데이터 13,000개 | 3 에포크 |
|-----|----------------|----------|
| T4 (16GB) | ~4시간 | batch=2 |
| A100 (40GB) | ~1.5시간 | batch=4 |
| RTX 4090 (24GB) | ~2시간 | batch=4 |

---

## 문제 해결

### OOM (Out of Memory)
```python
# 배치 크기 줄이기
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16

# 또는 시퀀스 길이 줄이기
MAX_SEQ_LENGTH = 1024
```

### Loss가 감소하지 않음
```python
# 학습률 조정
LEARNING_RATE = 5e-5  # 더 낮게

# 워밍업 늘리기
WARMUP_RATIO = 0.1
```

### 생성 품질이 낮음
- 학습 데이터 품질 검증
- 에포크 수 증가
- 더 큰 LoRA rank 시도 (r=64)

---

## 파일 구조

```
training/
├── finetune_html_ui_colab.ipynb  # Colab 노트북 (권장)
├── train_qlora_colab.py          # QLoRA 학습 스크립트
├── train_unsloth_colab.py        # Unsloth 학습 스크립트
└── README.md                      # 이 문서
```

---

## 다음 단계

1. 파인튜닝 완료 후 `models/` 폴더에 GGUF 파일 배치
2. `src/agent/generator.py`에서 모델 경로 설정
3. `python -m src.api.main`으로 API 서버 실행
