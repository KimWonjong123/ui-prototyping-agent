"""
Unsloth를 사용한 더 빠른 Fine-tuning
Unsloth는 QLoRA 대비 2배 빠르고 메모리 50% 절약

Google Colab Pro/T4에서도 원활하게 작동
"""

# ============================================================
# 1. Unsloth 설치 (Colab 첫 셀)
# ============================================================
"""
%%capture
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
"""

# ============================================================
# 2. 라이브러리 임포트
# ============================================================
import os
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ============================================================
# 3. 설정
# ============================================================
MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"  # 사전 양자화된 모델
OUTPUT_DIR = "./qwen-html-ui-unsloth"
MAX_SEQ_LENGTH = 2048

# LoRA 설정
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0

# 학습 설정 (Unsloth는 더 큰 배치 사용 가능)
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

# 학습 데이터 경로
TRAINING_DATA_PATH = "./training_data.jsonl"

# ============================================================
# 4. 모델 로드 (Unsloth 방식)
# ============================================================
def load_model_unsloth():
    """Unsloth로 4bit 모델 로드"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # 자동 감지
        load_in_4bit=True,
    )
    
    # LoRA 어댑터 추가
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth 최적화
        random_state=42,
    )
    
    return model, tokenizer

# ============================================================
# 5. 데이터 준비
# ============================================================
def load_and_prepare_data(tokenizer):
    """학습 데이터 로드 및 포맷팅"""
    
    # JSONL 로드
    data = []
    with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    
    print(f"총 {len(data)}개 데이터 로드")
    
    # 포맷팅 함수
    def format_prompt(example):
        messages = example['messages']
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}
    
    # Dataset 생성
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
    
    # 분리
    split = dataset.train_test_split(test_size=0.1, seed=42)
    
    return split['train'], split['test']

# ============================================================
# 6. 학습 실행
# ============================================================
def train():
    print("=" * 60)
    print("Unsloth Fine-tuning for HTML UI Generation")
    print("=" * 60)
    
    # 모델 로드
    print("\n[1/4] Unsloth 모델 로드 중...")
    model, tokenizer = load_model_unsloth()
    model.print_trainable_parameters()
    
    # 데이터 준비
    print("\n[2/4] 데이터 준비 중...")
    train_dataset, eval_dataset = load_and_prepare_data(tokenizer)
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # 학습 설정
    print("\n[3/4] 학습 설정 중...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
    )
    
    # 학습 시작
    print("\n[4/4] 학습 시작...")
    trainer.train()
    
    # 저장
    print("\n✅ 학습 완료! 모델 저장 중...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    return model, tokenizer

# ============================================================
# 7. GGUF 변환 (Unsloth 내장 기능)
# ============================================================
def export_to_gguf(model, tokenizer):
    """Unsloth의 내장 GGUF 변환 기능 사용"""
    print("\n[GGUF 변환] Q4_K_M 양자화로 내보내기...")
    
    # 다양한 양자화 옵션 가능:
    # q4_k_m, q5_k_m, q8_0, f16
    model.save_pretrained_gguf(
        "qwen-html-ui-gguf",
        tokenizer,
        quantization_method="q4_k_m"  # 로컬 추론용 최적
    )
    
    print("✅ GGUF 파일 생성 완료!")
    print("  - qwen-html-ui-gguf/unsloth.Q4_K_M.gguf")
    print("\n다음 단계: Ollama에 모델 등록")

# ============================================================
# 8. Ollama Modelfile 생성
# ============================================================
def create_ollama_modelfile():
    """Ollama용 Modelfile 생성"""
    modelfile_content = '''FROM ./qwen-html-ui-gguf/unsloth.Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048

SYSTEM """당신은 HTML/CSS UI 전문가입니다. 사용자의 요청에 따라 기존 디자인 시스템과 일관된 HTML 코드를 생성합니다.
- 시맨틱 HTML5 태그를 사용합니다
- 클래스명은 기존 컴포넌트 라이브러리의 규칙을 따릅니다
- 접근성(a11y)을 고려합니다
- 깔끔하고 유지보수하기 쉬운 코드를 작성합니다"""
'''
    
    with open("Modelfile", "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    
    print("\n✅ Modelfile 생성 완료!")
    print("\nOllama 등록 명령어:")
    print("  ollama create html-ui-agent -f Modelfile")
    print("\n사용 방법:")
    print("  ollama run html-ui-agent")

# ============================================================
# 9. 메인 실행
# ============================================================
if __name__ == "__main__":
    # GPU 확인
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 학습
    model, tokenizer = train()
    
    # GGUF 변환
    export_to_gguf(model, tokenizer)
    
    # Ollama Modelfile 생성
    create_ollama_modelfile()
