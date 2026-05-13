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
│   └── generate_training_data.py  # 메인 스크립트
│
└── src/
    └── utils/
        └── llm_client.py  # LLM 클라이언트 (Gemini/Ollama)
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
