# HTML/CSS 스타일 기반 UI 생성 AI Agent 프로젝트

## 📋 프로젝트 개요

기존 HTML/CSS 코드베이스를 기반으로, 텍스트 프롬프트 및 이미지 입력을 받아 **기존 스타일과 일관된 새로운 HTML을 생성**하는 로컬 AI Agent 시스템

### 핵심 목표
- "로그인 페이지 만들어줘" → 기존 디자인 시스템에 맞는 HTML 생성
- 외부 API 없이 **완전 로컬** 실행
- 기존 CSS 클래스 네이밍 규칙 준수

---

## 🔧 기술 스택 결정사항

### 추론 환경 제약
- **RAM:** 16GB
- **GPU:** 없음 (CPU only)
- **OS:** Windows

### 선정된 기술 스택

| 구성요소 | 선택 | 이유 |
|---------|------|------|
| **Code LLM** | Qwen2.5-Coder-7B-Q4_K_M | 코드 생성 최강, 16GB에서 구동 가능 |
| **VLM** | LLaVA-v1.6-7B-Q4 (선택적) | 이미지 입력 시에만 사용 |
| **Inference Engine** | Ollama (MVP) → llama.cpp (Production) | CPU 최적화, 설치 간편 |
| **Embedding** | jina-embeddings-v2-base-code 또는 nomic-embed-text | 코드 특화 |
| **Vector DB** | ChromaDB | 로컬 파일 기반, 설치 간편 |
| **Framework** | LangChain + FastAPI | RAG 구현 용이 |

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          User Interface                                  │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
│   │   Text Input    │    │  Image Upload   │    │  Output View    │    │
│   └────────┬────────┘    └────────┬────────┘    └────────▲────────┘    │
└────────────┼───────────────────────┼───────────────────────┼────────────┘
             │                       │                       │
             ▼                       ▼                       │
┌────────────────────────────────────────────────────────────┼────────────┐
│                      Input Processor                       │            │
│   ┌────────────────────┐    ┌────────────────────┐        │            │
│   │ Intent Parser      │    │ VLM (if image)     │        │            │
│   │ (lightweight)      │    │ LLaVA-7B-Q4        │        │            │
│   └─────────┬──────────┘    └─────────┬──────────┘        │            │
└─────────────┼──────────────────────────┼───────────────────┼────────────┘
              │                          │                   │
              ▼                          ▼                   │
┌─────────────────────────────────────────────────────────────────────────┐
│                      Context Builder                                     │
│   ┌────────────────────────────────────────────────────────────────┐    │
│   │                    RAG System                                   │    │
│   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │    │
│   │  │ Embedding   │    │ ChromaDB    │    │ Retriever   │        │    │
│   │  │ jina-code   │ →  │ (local)     │ →  │ top-k=5     │        │    │
│   │  └─────────────┘    └─────────────┘    └─────────────┘        │    │
│   └────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Generation Engine                                   │
│   ┌────────────────────────────────────────────────────────────────┐    │
│   │          Qwen2.5-Coder-7B-Q4_K_M (Fine-tuned + LoRA)           │    │
│   │                      via Ollama / llama.cpp                     │    │
│   └────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Post-Processing                                     │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
│   │ HTML Validator  │ →  │ CSS Class Check │ →  │ Preview Render  │    │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### VLM 사용 시 2단계 파이프라인

VLM은 Fine-tuning 없이 기본 모델 사용. Code LLM만 Fine-tuning.

```
이미지 입력 → VLM (UI 설명 생성) → Code LLM (HTML 생성)
             "눈" 역할              "손" 역할 (Fine-tuned)
```

---

## 📊 학습 데이터 준비

### 데이터 소스
- 기존 시스템의 HTML 파일들
- HTML에 Tailwind-like 클래스 기반 스타일링 적용됨
- 기본 CSS 템플릿은 학습 불필요, 커스텀 CSS만 선택적 포함

### 데이터 포맷: ChatML

```json
{
  "messages": [
    {"role": "system", "content": "당신은 HTML/CSS 전문가입니다..."},
    {"role": "user", "content": "로그인 폼을 만들어줘"},
    {"role": "assistant", "content": "<form class='form-container'>...</form>"}
  ]
}
```

### 데이터 준비 전략

#### 방법 1: 전체 페이지 단위 (간단, 빠른 시작)
```python
# 파일명에서 페이지 유형 추론 → instruction 생성
"login.html" → "로그인 페이지를 만들어줘"
```

#### 방법 2: 컴포넌트 단위 (품질 높음)
```python
# HTML에서 form, card, button 등 추출 → 개별 학습 데이터
```

#### 권장: 혼합 접근법
| 데이터 유형 | 비율 | 용도 |
|------------|------|------|
| 전체 페이지 | 20-30% | "~페이지 만들어줘" |
| 큰 섹션 | 20-30% | "헤더 영역", "푸터 영역" |
| 개별 컴포넌트 | 40-60% | "버튼", "카드", "폼" |

### Instruction 자동 생성
Ollama + LLM을 사용하여 HTML에 맞는 자연스러운 요청문 자동 생성

```bash
ollama pull qwen2.5-coder:7b
# HTML → LLM → "이 HTML을 만들어달라고 요청하는 문장 3개 생성"
```

---

## 🎓 Fine-tuning 전략

### 학습-추론 분리
- **학습:** 클라우드 GPU (A100 40GB 권장)
- **추론:** 로컬 CPU (16GB RAM)

### Fine-tuning 방식: QLoRA

```python
# 핵심 하이퍼파라미터
LORA_R = 32           # rank
LORA_ALPHA = 64       # scaling
LEARNING_RATE = 2e-4
EPOCHS = 3
MAX_SEQ_LENGTH = 2048
```

### 학습 후 처리
1. LoRA 어댑터 병합 (merge)
2. GGUF 변환
3. Q4_K_M 양자화

```bash
# 최종 결과물
model-Q4_K_M.gguf (~4.5GB)
```

---

## 📁 프로젝트 구조 (권장)

```
html-ui-agent/
├── AGENT_GUIDE.md              # 이 파일
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw_html/               # 원본 HTML 파일들
│   ├── extracted/              # 추출된 컴포넌트
│   └── training/               # 학습 데이터 (JSON)
│
├── scripts/
│   ├── extract_components.py   # HTML → 컴포넌트 추출
│   ├── generate_instructions.py # LLM으로 instruction 생성
│   └── prepare_dataset.py      # 최종 학습 데이터 생성
│
├── training/
│   ├── train_lora.py           # QLoRA 학습 스크립트
│   ├── merge_lora.py           # LoRA 병합
│   └── quantize.sh             # GGUF 양자화
│
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── generator.py        # HTML 생성 로직
│   │   ├── rag.py              # RAG 시스템
│   │   └── vlm.py              # VLM 연동 (선택적)
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py             # FastAPI 서버
│   │
│   └── utils/
│       ├── html_parser.py
│       └── validators.py
│
├── models/                     # 로컬 모델 파일
│   └── .gitkeep
│
└── tests/
    └── test_generator.py
```

---

## 🚀 개발 로드맵

### Phase 1: MVP (2-4주)
- [ ] HTML 파일에서 학습 데이터 추출 파이프라인
- [ ] Ollama로 기본 생성 테스트 (Fine-tuning 전)
- [ ] 간단한 RAG 구현 (ChromaDB + jina-embeddings)
- [ ] Gradio 데모 UI

### Phase 2: Fine-tuning (4-8주)
- [ ] 학습 데이터 품질 검증 및 증강
- [ ] 클라우드에서 QLoRA 학습
- [ ] GGUF 양자화 및 로컬 테스트
- [ ] 성능 평가 및 개선

### Phase 3: Production (8-16주)
- [ ] VLM 통합 (이미지 입력)
- [ ] FastAPI 서버 구축
- [ ] HTML 검증 및 프리뷰
- [ ] 에러 처리 및 안정화

---

## 💡 핵심 의사결정 사항

### Q: RAG만으로 충분한가?
**A:** 단순 컴포넌트는 RAG만으로 가능하나, 스타일 일관성을 위해 Fine-tuning 권장

### Q: 스타일 일관성은 retrieval 문제인가, generation 문제인가?
**A:** 둘 다이지만 generation이 더 중요. Fine-tuned 모델이 스타일 "감각"을 가져야 함

### Q: VLM도 Fine-tuning 필요한가?
**A:** 대부분 불필요. VLM은 "설명 생성", Code LLM은 "코드 생성"으로 역할 분리

### Q: HTML만으로 학습 충분한가?
**A:** 클래스 네이밍이 Semantic하면 (btn-primary, card-lg 등) HTML만으로 충분

---

## 📚 참고 자료

### 모델
- [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [jina-embeddings-v2-base-code](https://huggingface.co/jinaai/jina-embeddings-v2-base-code)

### 도구
- [Ollama](https://ollama.com)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [ChromaDB](https://trychroma.com)
- [LangChain](https://github.com/langchain-ai/langchain)

### 논문
- Design2Code (arXiv:2403.03163) - Screenshot → Code 벤치마크
- LLaVA-1.5 (arXiv:2310.03744) - Visual Instruction Tuning

---

## ⚠️ 주의사항

### 메모리 관리
- 7B 모델 Q4 양자화 → ~5GB RAM 사용
- VLM 사용 시 순차 실행 필요 (동시 로드 X)

### 학습 데이터 품질
- HTML 태그 정상 닫힘 확인
- 클래스명 일관성 검증
- 너무 짧거나 긴 예제 제거 (50-5000자)

### 예상 병목
- 스타일 불일치 → 더 많은 학습 데이터, 증강
- 느린 생성 → Speculative decoding 고려
- Complex layout 실패 → 템플릿 기반 접근

---

## 🔑 환경 변수 및 설정

```bash
# .env 예시
OLLAMA_HOST=http://localhost:11434
CHROMA_DB_PATH=./data/chromadb
MODEL_PATH=./models/qwen2.5-coder-7b-q4.gguf
EMBEDDING_MODEL=jinaai/jina-embeddings-v2-base-code
```

---

*이 문서는 프로젝트 진행에 따라 업데이트됩니다.*
