"""
QLoRA Fine-tuning Script for Qwen2.5-Coder-7B
Google Colab Pro (A100/V100) 또는 클라우드 GPU에서 실행

사용법:
1. Google Colab에서 이 파일을 업로드
2. training_data.jsonl 파일도 업로드
3. 셀 단위로 실행

필요 환경: 
- GPU: A100 40GB 권장 (T4 16GB도 가능하나 batch_size 조정 필요)
- Python 3.10+
"""

# ============================================================
# 1. 필수 패키지 설치 (Colab 첫 셀에서 실행)
# ============================================================
"""
!pip install -q torch torchvision torchaudio
!pip install -q transformers==4.44.0
!pip install -q datasets==2.19.0
!pip install -q accelerate==0.30.0
!pip install -q peft==0.11.0
!pip install -q bitsandbytes==0.43.0
!pip install -q trl==0.8.6
!pip install -q wandb  # 학습 모니터링 (선택)
"""

# ============================================================
# 2. 라이브러리 임포트
# ============================================================
import os
import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ============================================================
# 3. 설정
# ============================================================
# 모델 설정
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
OUTPUT_DIR = "./qwen-html-ui-lora"
FINAL_MODEL_DIR = "./qwen-html-ui-merged"

# 학습 데이터 경로
TRAINING_DATA_PATH = "./training_data.jsonl"  # Colab에 업로드된 경로

# QLoRA 하이퍼파라미터
LORA_R = 32                # LoRA rank
LORA_ALPHA = 64            # LoRA alpha (보통 rank의 2배)
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 학습 하이퍼파라미터
BATCH_SIZE = 4             # A100: 4, T4: 1-2
GRADIENT_ACCUMULATION = 4  # 효과적인 배치 크기 = BATCH_SIZE * GRADIENT_ACCUMULATION
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 2048
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 100

# ============================================================
# 4. 데이터 로드 및 전처리
# ============================================================
def load_training_data(file_path):
    """JSONL 파일에서 학습 데이터 로드"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data

def format_chat_template(example, tokenizer):
    """ChatML 형식을 Qwen 형식으로 변환"""
    messages = example['messages']
    
    # Qwen의 apply_chat_template 사용
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

# ============================================================
# 5. 모델 및 토크나이저 로드
# ============================================================
def load_model_and_tokenizer():
    """4bit 양자화된 모델과 토크나이저 로드"""
    
    # 4bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # pad_token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

# ============================================================
# 6. LoRA 설정 및 적용
# ============================================================
def setup_lora(model):
    """LoRA 어댑터 설정"""
    
    # gradient checkpointing 활성화
    model.gradient_checkpointing_enable()
    
    # k-bit 학습 준비
    model = prepare_model_for_kbit_training(model)
    
    # LoRA 설정
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # LoRA 적용
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

# ============================================================
# 7. 학습 실행
# ============================================================
def train():
    print("=" * 60)
    print("QLoRA Fine-tuning for HTML UI Generation")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n[1/5] 학습 데이터 로드 중...")
    raw_data = load_training_data(TRAINING_DATA_PATH)
    print(f"  - 총 {len(raw_data)}개 샘플 로드됨")
    
    # 2. 모델 로드
    print("\n[2/5] 모델 및 토크나이저 로드 중...")
    model, tokenizer = load_model_and_tokenizer()
    
    # 3. 데이터셋 처리
    print("\n[3/5] 데이터셋 전처리 중...")
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset.column_names
    )
    print(f"  - 전처리 완료: {len(dataset)}개 샘플")
    
    # Train/Validation 분리 (90/10)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    print(f"  - Train: {len(train_dataset)}, Validation: {len(eval_dataset)}")
    
    # 4. LoRA 설정
    print("\n[4/5] LoRA 어댑터 설정 중...")
    model = setup_lora(model)
    
    # 5. 학습 설정
    print("\n[5/5] 학습 시작...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=SAVE_STEPS,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        group_by_length=True,
        report_to="none",  # wandb 사용 시 "wandb"로 변경
    )
    
    # Trainer 설정
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
    
    # 학습 실행
    trainer.train()
    
    # 모델 저장
    print("\n학습 완료! 모델 저장 중...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\n✅ LoRA 어댑터 저장 완료: {OUTPUT_DIR}")
    return model, tokenizer

# ============================================================
# 8. LoRA 병합 및 저장
# ============================================================
def merge_and_save(model, tokenizer):
    """LoRA를 기본 모델에 병합하고 저장"""
    print("\n[추가] LoRA 어댑터를 기본 모델에 병합 중...")
    
    # LoRA 병합
    merged_model = model.merge_and_unload()
    
    # 병합된 모델 저장
    merged_model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    
    print(f"✅ 병합된 모델 저장 완료: {FINAL_MODEL_DIR}")
    print("\n다음 단계: GGUF 변환 후 Ollama에서 사용")

# ============================================================
# 9. 메인 실행
# ============================================================
if __name__ == "__main__":
    # GPU 확인
    if not torch.cuda.is_available():
        print("⚠️ GPU를 찾을 수 없습니다. Colab에서 GPU 런타임을 선택하세요.")
        print("  런타임 > 런타임 유형 변경 > GPU 선택")
    else:
        print(f"✅ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"  - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 학습 실행
        model, tokenizer = train()
        
        # LoRA 병합 (선택적)
        # merge_and_save(model, tokenizer)
