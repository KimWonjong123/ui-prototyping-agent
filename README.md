# UI Prototyping Agent - Training Data Generator

HTML/CSS 코드베이스에서 파인튜닝용 학습 데이터를 자동 생성하는 도구입니다.

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env.example을 복사하여 .env 파일 생성
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac
```

`.env` 파일을 편집하여 설정:

**Gemini 사용 시:**
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-api-key-here
GEMINI_MODEL=gemini-1.5-flash
```

**Ollama 사용 시:**
```env
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

### 3. HTML 파일 준비

```bash
# data/raw_html 디렉토리에 HTML 파일 추가
data/
└── raw_html/
    ├── login.html
    ├── dashboard.html
    └── components/
        ├── button.html
        └── card.html
```

### 4. 학습 데이터 생성

```bash
# 기본 실행
python scripts/generate_training_data.py

# 옵션 지정
python scripts/generate_training_data.py \
    --input data/raw_html \
    --output data/training \
    --provider gemini \
    --language ko \
    --num-instructions 3
```

### 5. 배치 처리로 빠르게! (선택)

**ollama 추론이 느리면 배치 처리를 사용하세요:**

```bash
# 배치 사이즈 지정으로 더 빠르게
python scripts/generate_training_data.py \
    --batch-size 8 \
    --input data/raw_html \
    --output data/training

# 배치 처리 유틸리티 사용 (더 세밀한 제어)
python scripts/batch_inference.py \
    --input data/extracted_components.json \
    --batch-size 8 \
    --output results.json

# 벤치마크로 최적 배치 사이즈 찾기
python scripts/batch_inference.py --benchmark --batch-size 4 --num-items 100
```

**배치 사이즈 가이드:**
- `--batch-size 1`: 느림 (하나씩 처리)
- `--batch-size 4`: 권장 (기본값, 균형잡힘)
- `--batch-size 8-16`: 빠름 (메모리 넉넉할 때)
- `--batch-size 32+`: 매우 빠름 (고사양 필요)

## 📁 프로젝트 구조

```
ui-prototyping-agent/
├── .env.example          # 환경 변수 템플릿
├── requirements.txt      # Python 패키지
├── AGENTS.md            # 프로젝트 가이드
│
├── data/
│   ├── raw_html/        # 입력: 원본 HTML 파일
│   ├── extracted/       # 추출된 컴포넌트
│   └── training/        # 출력: 학습 데이터 (JSON/JSONL)
│
├── scripts/
│   ├── generate_training_data.py  # 메인 스크립트 (배치 지원)
│   └── batch_inference.py         # 배치 처리 유틸리티
│
└── src/
    └── utils/
        └── llm_client.py  # LLM 클라이언트 (Gemini/Ollama + 배치 처리)
```

## 🔧 CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--input`, `-i` | HTML 파일 디렉토리 | `data/raw_html` |
| `--output`, `-o` | 출력 디렉토리 | `data/training` |
| `--provider`, `-p` | LLM 제공자 (`gemini` / `ollama`) | `.env`에서 |
| `--model`, `-m` | 모델 이름 | `.env`에서 |
| `--language`, `-l` | 생성 언어 (`ko` / `en`) | `ko` |
| `--num-instructions`, `-n` | 컴포넌트당 instruction 수 | `3` |
| `--categories`, `-c` | 추출 카테고리 | `page,section,component` |
| `--batch-size`, `-b` | 배치 처리 크기 (빠른 추론!) | `4` |
| `--test` | LLM 연결 테스트 | - |

## 📊 출력 포맷

### ChatML (JSON)
```json
{
  "messages": [
    {"role": "system", "content": "당신은 HTML/CSS UI 전문가입니다..."},
    {"role": "user", "content": "로그인 폼을 만들어줘"},
    {"role": "assistant", "content": "<form class='login-form'>...</form>"}
  ],
  "metadata": {
    "source": "data/raw_html/login.html",
    "type": "component",
    "tag": "form"
  }
}
```

### JSONL (라인 단위)
```jsonl
{"messages": [...], "metadata": {...}}
{"messages": [...], "metadata": {...}}
```

## 🚀 배치 처리 (Batch Inference)

**Ollama 추론 속도를 크게 향상시키는 배치 처리 기능입니다!**

### 배치 처리란?

- 여러 프롬프트를 한 번에 LLM에 전송 (네트워크 오버헤드 감소)
- 로컬 Ollama는 순차 처리하지만, 바뀌는 오버헤드 최소화
- 결과적으로 시간 단축 (보통 20-40% 빠름)

### 사용 방법

**1. generate_training_data.py에서 배치 처리**

```bash
# 배치 사이즈 지정
python scripts/generate_training_data.py --batch-size 8

# 환경 변수로도 설정 가능
BATCH_SIZE=8 python scripts/generate_training_data.py
```

**2. batch_inference.py로 더 세밀한 제어**

```bash
# 추출된 컴포넌트로 배치 처리
python scripts/batch_inference.py \
    --input data/extracted_components.json \
    --batch-size 8 \
    --output results.json

# JSONL 형식 지원
python scripts/batch_inference.py \
    --input components.jsonl \
    --output results.jsonl \
    --format jsonl

# 커스텀 프롬프트 배치 처리
python scripts/batch_inference.py \
    --prompts prompts.txt \
    --batch-size 4
```

**3. 성능 벤치마크**

```bash
# 최적 배치 사이즈 찾기
python scripts/batch_inference.py \
    --benchmark \
    --num-items 100 \
    --batch-size 4

# 출력:
# Benchmark Results
# ============================================================
# num_items: 100
# batch_size: 4
# total_time_seconds: 42.5000
# time_per_item_seconds: 0.4250
# items_per_second: 2.3529
# prompt_length: 100
# avg_response_length: 245.5000
# ============================================================
```

### 배치 사이즈 권장값

| 배치 사이즈 | 속도 | 메모리 | 추천 상황 |
|----------|------|--------|---------|
| 1 | 느림 | 낮음 | 테스트용 |
| 2-4 | 중간 | 중간 | **기본 권장** |
| 4-8 | 빠름 | 중간-높음 | 데이터 많을 때 |
| 8-16 | 매우 빠름 | 높음 | 고사양, 데이터 많음 |
| 16+ | 초고속 | 매우 높음 | 워크스테이션급 |

> **팁:** 벤치마크로 자신의 시스템 최적값을 찾아서 사용하세요!

## 🤖 LLM 제공자 설정

### Gemini (Google AI)

1. [Google AI Studio](https://aistudio.google.com/)에서 API 키 발급
2. `.env`에 `GEMINI_API_KEY` 설정
3. 권장 모델: `gemini-1.5-flash` (빠름, 저렴) 또는 `gemini-1.5-pro` (고품질)

### Ollama (로컬)

1. [Ollama](https://ollama.com/) 설치
2. 모델 다운로드:
   ```bash
   ollama pull qwen2.5:7b
   ```
3. Ollama 서버 실행 (자동으로 실행됨)

**권장 로컬 모델:**
| 모델 | 크기 | 특징 |
|------|------|------|
| `qwen2.5:7b` | ~4.4GB | 한국어+코드 우수, **추천** |
| `llama3.1:8b` | ~4.7GB | 범용 성능 우수 |
| `mistral:7b` | ~4.1GB | 빠른 속도 |

## ⚙️ 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `LLM_PROVIDER` | `gemini` 또는 `ollama` | `gemini` |
| `GEMINI_API_KEY` | Gemini API 키 | - |
| `GEMINI_MODEL` | Gemini 모델명 | `gemini-1.5-flash` |
| `OLLAMA_HOST` | Ollama 서버 주소 | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama 모델명 | `qwen2.5:7b` |
| `NUM_INSTRUCTIONS_PER_COMPONENT` | 컴포넌트당 instruction 수 | `3` |
| `INSTRUCTION_LANGUAGE` | 생성 언어 | `ko` |
| `EXTRACT_CATEGORIES` | 추출 카테고리 | `page,section,component` |
| `MIN_HTML_LENGTH` | 최소 HTML 길이 | `50` |
| `MAX_HTML_LENGTH` | 최대 HTML 길이 | `10000` |

## 📝 예시

### LLM 연결 테스트
```bash
python scripts/generate_training_data.py --test
```

### Gemini로 학습 데이터 생성
```bash
python scripts/generate_training_data.py -p gemini -n 5 -l ko
```

### Ollama로 학습 데이터 생성
```bash
# Ollama 서버가 실행 중인지 확인
ollama list

# 학습 데이터 생성
python scripts/generate_training_data.py -p ollama -m qwen2.5:7b
```

### 컴포넌트만 추출
```bash
python scripts/generate_training_data.py --categories component
```

## 🔍 HTML 추출 기준

스크립트는 다음 기준으로 HTML 컴포넌트를 자동 추출합니다:

**페이지 (`page`):**
- `<body>` 태그 전체 (길이 제한 내)

**섹션 (`section`):**
- `<header>`, `<footer>`, `<main>`, `<aside>`
- `<section>`, `<article>`

**컴포넌트 (`component`):**
- `<form>`, `<nav>`, `<table>`, `<ul>`, `<ol>`
- `<dialog>`, `<figure>`
- 특정 클래스 패턴을 가진 `<div>`:
  - `card`, `modal`, `dropdown`, `menu`, `tab`
  - `btn`, `button`, `form`, `alert`, `toast` 등

## ❓ 트러블슈팅

### Gemini API 오류
```
Error: GEMINI_API_KEY is required
```
→ `.env` 파일에 유효한 API 키 설정 필요

### Ollama 연결 실패
```
Ollama connection failed: Connection refused
```
→ Ollama 서버가 실행 중인지 확인: `ollama serve`

### 모델을 찾을 수 없음
```
Warning: Model 'qwen2.5:7b' not found
```
→ 모델 다운로드: `ollama pull qwen2.5:7b`

### 인코딩 오류
스크립트는 UTF-8, CP949, EUC-KR 인코딩을 자동으로 시도합니다.

---

## 라이선스

MIT License
