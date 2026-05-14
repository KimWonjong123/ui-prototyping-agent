"""
Hugging Face 모델을 GGUF로 변환하는 스크립트
llama.cpp 필요

사용법:
1. llama.cpp 설치
2. 병합된 모델 경로 지정
3. 스크립트 실행
"""

import os
import subprocess
import argparse

def convert_to_gguf(model_path: str, output_dir: str, quantization: str = "q4_k_m"):
    """
    HuggingFace 모델을 GGUF 형식으로 변환
    
    Args:
        model_path: 병합된 HuggingFace 모델 경로
        output_dir: GGUF 파일 출력 디렉토리
        quantization: 양자화 방식 (q4_k_m, q5_k_m, q8_0, f16)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # llama.cpp 경로 (설치 필요)
    LLAMA_CPP_PATH = os.environ.get("LLAMA_CPP_PATH", "./llama.cpp")
    
    print(f"[1/3] Converting model to GGUF F16...")
    
    # Step 1: HF to GGUF (F16)
    f16_path = os.path.join(output_dir, "model-f16.gguf")
    convert_cmd = [
        "python", f"{LLAMA_CPP_PATH}/convert_hf_to_gguf.py",
        model_path,
        "--outfile", f16_path,
        "--outtype", "f16"
    ]
    
    try:
        subprocess.run(convert_cmd, check=True)
        print(f"  ✅ F16 GGUF 생성: {f16_path}")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ 변환 실패: {e}")
        return None
    
    # Step 2: Quantize
    print(f"\n[2/3] Quantizing to {quantization}...")
    
    quant_path = os.path.join(output_dir, f"model-{quantization}.gguf")
    quantize_cmd = [
        f"{LLAMA_CPP_PATH}/build/bin/llama-quantize",
        f16_path,
        quant_path,
        quantization.upper()
    ]
    
    try:
        subprocess.run(quantize_cmd, check=True)
        print(f"  ✅ 양자화 완료: {quant_path}")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ 양자화 실패: {e}")
        return None
    
    # Step 3: Modelfile 생성
    print(f"\n[3/3] Creating Ollama Modelfile...")
    
    modelfile_content = f'''FROM ./{os.path.basename(quant_path)}

TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
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
    
    modelfile_path = os.path.join(output_dir, "Modelfile")
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    
    print(f"  ✅ Modelfile 생성: {modelfile_path}")
    
    # 안내
    print("\n" + "=" * 60)
    print("변환 완료!")
    print("=" * 60)
    print(f"\nOllama에 등록하려면:")
    print(f"  cd {output_dir}")
    print(f"  ollama create html-ui-agent -f Modelfile")
    print(f"\n실행:")
    print(f"  ollama run html-ui-agent")
    
    return quant_path


def main():
    parser = argparse.ArgumentParser(description="Convert HF model to GGUF for Ollama")
    parser.add_argument("--model", required=True, help="Path to merged HF model")
    parser.add_argument("--output", default="./models/gguf", help="Output directory")
    parser.add_argument("--quant", default="q4_k_m", 
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="Quantization method")
    
    args = parser.parse_args()
    convert_to_gguf(args.model, args.output, args.quant)


if __name__ == "__main__":
    main()
