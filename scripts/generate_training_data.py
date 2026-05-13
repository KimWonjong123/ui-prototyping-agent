#!/usr/bin/env python
"""
Training Data Generator for HTML UI Agent

This script generates fine-tuning training data from HTML files by:
1. Extracting HTML components (pages, sections, components)
2. Using LLM to generate natural language instructions for each component
3. Saving in ChatML format for fine-tuning

Usage:
    python scripts/generate_training_data.py --input data/raw_html --output data/training
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.llm_client import get_llm_client, LLMClient

# Load environment variables
load_dotenv()


# ============================================
# Configuration
# ============================================

DEFAULT_SYSTEM_PROMPT = """당신은 HTML/CSS UI 전문가입니다. 사용자의 요청에 따라 기존 디자인 시스템과 일관된 HTML 코드를 생성합니다.
- 시맨틱 HTML5 태그를 사용합니다
- 클래스명은 기존 컴포넌트 라이브러리의 규칙을 따릅니다
- 접근성(a11y)을 고려합니다
- 깔끔하고 유지보수하기 쉬운 코드를 작성합니다"""

INSTRUCTION_GENERATION_PROMPT_KO = """다음 HTML 코드를 보고, 이 HTML을 만들어달라고 요청하는 자연스러운 한국어 문장을 {num_instructions}개 생성해주세요.

요청문은 다양한 스타일로 작성해주세요:
- 간단한 요청 (예: "로그인 폼 만들어줘")
- 상세한 요청 (예: "이메일과 비밀번호 입력 필드가 있는 로그인 폼을 만들어줘")
- 기능 중심 요청 (예: "사용자 인증을 위한 로그인 화면이 필요해")

컴포넌트 유형: {component_type}

HTML 코드:
```html
{html_content}
```

JSON 배열 형식으로만 응답해주세요. 다른 설명은 필요 없습니다:
["요청문1", "요청문2", "요청문3"]"""

INSTRUCTION_GENERATION_PROMPT_EN = """Look at the following HTML code and generate {num_instructions} natural language instructions that would request creating this HTML.

Write instructions in various styles:
- Simple requests (e.g., "Create a login form")
- Detailed requests (e.g., "Create a login form with email and password input fields")
- Function-focused requests (e.g., "I need a login screen for user authentication")

Component type: {component_type}

HTML code:
```html
{html_content}
```

Respond only in JSON array format. No other explanation needed:
["instruction1", "instruction2", "instruction3"]"""


# ============================================
# HTML Extraction
# ============================================

class HTMLExtractor:
    """Extract components from HTML files"""
    
    # Tags that typically represent standalone components
    COMPONENT_TAGS = {
        'form', 'nav', 'header', 'footer', 'aside', 'main', 'article',
        'section', 'dialog', 'table', 'ul', 'ol', 'figure'
    }
    
    # Class patterns that indicate a component
    COMPONENT_CLASS_PATTERNS = [
        r'card', r'modal', r'dropdown', r'menu', r'tab', r'accordion',
        r'carousel', r'slider', r'alert', r'toast', r'badge', r'chip',
        r'avatar', r'btn', r'button', r'input', r'form', r'search',
        r'pagination', r'breadcrumb', r'progress', r'spinner', r'skeleton'
    ]
    
    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 10000
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.component_pattern = re.compile(
            '|'.join(self.COMPONENT_CLASS_PATTERNS),
            re.IGNORECASE
        )
    
    def extract_from_file(self, file_path: Path) -> list[dict[str, Any]]:
        """
        Extract components from an HTML file
        
        Returns list of dicts with:
            - content: HTML string
            - type: 'page', 'section', or 'component'
            - source: original file path
            - tag: HTML tag name
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['cp949', 'euc-kr', 'latin-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        html_content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print(f"Warning: Could not read {file_path} with any encoding")
                return []
        
        results = []
        soup = BeautifulSoup(html_content, 'lxml')
        
        # 1. Extract full page (if within length limits)
        body = soup.find('body')
        if body:
            body_html = str(body)
            if self.min_length <= len(body_html) <= self.max_length:
                results.append({
                    'content': self._clean_html(body_html),
                    'type': 'page',
                    'source': str(file_path),
                    'tag': 'body'
                })
        
        # 2. Extract sections and components
        for tag in self.COMPONENT_TAGS:
            for element in soup.find_all(tag):
                element_html = str(element)
                
                if not (self.min_length <= len(element_html) <= self.max_length):
                    continue
                
                component_type = self._classify_component(element, tag)
                
                results.append({
                    'content': self._clean_html(element_html),
                    'type': component_type,
                    'source': str(file_path),
                    'tag': tag
                })
        
        # 3. Extract div/span components with meaningful classes
        for element in soup.find_all(['div', 'span']):
            classes = element.get('class', [])
            if not classes:
                continue
            
            class_str = ' '.join(classes)
            if not self.component_pattern.search(class_str):
                continue
            
            element_html = str(element)
            if not (self.min_length <= len(element_html) <= self.max_length):
                continue
            
            results.append({
                'content': self._clean_html(element_html),
                'type': 'component',
                'source': str(file_path),
                'tag': element.name
            })
        
        return results
    
    def _classify_component(self, element, tag: str) -> str:
        """Classify component as page, section, or component"""
        if tag in {'header', 'footer', 'main', 'aside'}:
            return 'section'
        elif tag in {'section', 'article'}:
            return 'section'
        else:
            return 'component'
    
    def _clean_html(self, html: str) -> str:
        """Clean and normalize HTML content"""
        # Remove excessive whitespace while preserving structure
        html = re.sub(r'\n\s*\n', '\n', html)
        html = re.sub(r'  +', ' ', html)
        return html.strip()


# ============================================
# Instruction Generation
# ============================================

class InstructionGenerator:
    """Generate natural language instructions using LLM"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        language: str = 'ko',
        num_instructions: int = 3
    ):
        self.llm = llm_client
        self.language = language
        self.num_instructions = num_instructions
        
        self.prompt_template = (
            INSTRUCTION_GENERATION_PROMPT_KO if language == 'ko'
            else INSTRUCTION_GENERATION_PROMPT_EN
        )
    
    def generate(self, html_content: str, component_type: str) -> list[str]:
        """Generate instructions for the given HTML content"""
        prompt = self.prompt_template.format(
            html_content=html_content[:5000],  # Truncate long content
            component_type=component_type,
            num_instructions=self.num_instructions
        )
        
        try:
            response = self.llm.generate(prompt)
            instructions = self._parse_response(response)
            
            if not instructions:
                # Fallback: generate simple instruction from component type
                instructions = [self._generate_fallback_instruction(component_type)]
            
            return instructions
        except Exception as e:
            print(f"Warning: LLM generation failed: {e}")
            return [self._generate_fallback_instruction(component_type)]
    
    def generate_batch(
        self,
        components: list[dict],
        batch_size: int = 4,
        show_progress: bool = True
    ) -> list[list[str]]:
        """
        Generate instructions for multiple components in batches
        
        Args:
            components: List of dicts with 'content' and 'type' keys
            batch_size: Number of prompts to process between status updates
            show_progress: Whether to print progress information
        
        Returns:
            List of instruction lists, one per component in the same order
        """
        all_results = []
        
        # Build all prompts first
        prompts = []
        component_types = []
        for component in components:
            prompt = self.prompt_template.format(
                html_content=component['content'][:5000],
                component_type=component['type'],
                num_instructions=self.num_instructions
            )
            prompts.append(prompt)
            component_types.append(component['type'])
        
        if show_progress:
            print(f"Generating instructions for {len(prompts)} components (batch size: {batch_size})")
        
        # Use LLM's batch method if available, otherwise fall back to sequential
        try:
            responses = self.llm.generate_batch(prompts, batch_size=batch_size)
        except (AttributeError, NotImplementedError):
            # Fallback for LLM clients that don't support batch
            if show_progress:
                print("LLM provider doesn't support batch processing, falling back to sequential...")
            responses = []
            iterator = tqdm(prompts, desc="Generating instructions", unit="component") if show_progress else prompts
            for prompt in iterator:
                try:
                    response = self.llm.generate(prompt)
                    responses.append(response)
                except Exception as e:
                    if show_progress:
                        tqdm.write(f"Warning: Generation failed: {e}")
                    responses.append("")
        
        # Parse responses
        iterator = tqdm(
            zip(responses, component_types),
            total=len(responses),
            desc="Parsing responses",
            unit="response"
        ) if show_progress else zip(responses, component_types)
        
        for response, component_type in iterator:
            try:
                instructions = self._parse_response(response)
                if not instructions:
                    instructions = [self._generate_fallback_instruction(component_type)]
            except Exception as e:
                if show_progress:
                    tqdm.write(f"Warning: Failed to parse response: {e}")
                instructions = [self._generate_fallback_instruction(component_type)]
            
            all_results.append(instructions)
        
        if show_progress:
            print(f"✓ Generated {len(all_results)} instruction sets")
        
        return all_results
    
    def _parse_response(self, response: str) -> list[str]:
        """Parse JSON array from LLM response"""
        # Try to extract JSON array from response
        try:
            # Find JSON array in response
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass
        
        # Try parsing entire response as JSON
        try:
            result = json.loads(response)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        
        return []
    
    def _generate_fallback_instruction(self, component_type: str) -> str:
        """Generate a simple fallback instruction"""
        if self.language == 'ko':
            type_map = {
                'page': '페이지',
                'section': '섹션',
                'component': '컴포넌트'
            }
            return f"{type_map.get(component_type, '컴포넌트')}를 만들어줘"
        else:
            return f"Create a {component_type}"


# ============================================
# Training Data Builder
# ============================================

class TrainingDataBuilder:
    """Build ChatML format training data"""
    
    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.system_prompt = system_prompt
    
    def build(
        self,
        instruction: str,
        html_content: str,
        metadata: Optional[dict] = None
    ) -> dict[str, Any]:
        """
        Build a single training example in ChatML format
        """
        example = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": html_content}
            ]
        }
        
        if metadata:
            example["metadata"] = metadata
        
        return example


# ============================================
# Main Pipeline
# ============================================

class TrainingDataPipeline:
    """Main pipeline for generating training data"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        language: str = 'ko',
        num_instructions: int = 3,
        min_html_length: int = 50,
        max_html_length: int = 10000,
        extract_categories: list[str] = None,
        batch_size: int = 4
    ):
        self.extractor = HTMLExtractor(min_html_length, max_html_length)
        self.instruction_gen = InstructionGenerator(llm_client, language, num_instructions)
        self.builder = TrainingDataBuilder()
        self.extract_categories = extract_categories or ['page', 'section', 'component']
        self.batch_size = batch_size
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        save_intermediate: bool = True
    ) -> list[dict]:
        """
        Process all HTML files in a directory
        
        Args:
            input_dir: Directory containing HTML files
            output_dir: Directory to save training data
            save_intermediate: Whether to save extracted components separately
        
        Returns:
            List of training examples
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all HTML files (supports .html, .htm, .div extensions)
        html_files = (
            list(input_dir.glob('**/*.html')) + 
            list(input_dir.glob('**/*.htm')) +
            list(input_dir.glob('**/*.div'))
        )
        
        if not html_files:
            print(f"No HTML/DIV files found in {input_dir}")
            return []
        
        print(f"Found {len(html_files)} HTML files")
        
        # Extract components
        all_components = []
        for html_file in tqdm(html_files, desc="Extracting components"):
            components = self.extractor.extract_from_file(html_file)
            # Filter by categories
            components = [c for c in components if c['type'] in self.extract_categories]
            all_components.extend(components)
        
        print(f"Extracted {len(all_components)} components")
        
        if save_intermediate:
            extracted_path = output_dir / 'extracted_components.json'
            with open(extracted_path, 'w', encoding='utf-8') as f:
                json.dump(all_components, f, ensure_ascii=False, indent=2)
            print(f"Saved extracted components to {extracted_path}")
        
        # Generate instructions and build training data using batch processing
        print(f"\nGenerating instructions for {len(all_components)} components...")
        print(f"Batch size: {self.batch_size}")
        
        all_instruction_sets = self.instruction_gen.generate_batch(
            all_components,
            batch_size=self.batch_size,
            show_progress=True
        )
        
        training_data = []
        for component, instructions in tqdm(
            zip(all_components, all_instruction_sets),
            total=len(all_components),
            desc="Building training examples"
        ):
            for instruction in instructions:
                example = self.builder.build(
                    instruction=instruction,
                    html_content=component['content'],
                    metadata={
                        'source': component['source'],
                        'type': component['type'],
                        'tag': component['tag']
                    }
                )
                training_data.append(example)
        
        print(f"Generated {len(training_data)} training examples")
        
        # Save training data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f'training_data_{timestamp}.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved training data to {output_path}")
        
        # Also save in JSONL format (one example per line, better for training)
        jsonl_path = output_dir / f'training_data_{timestamp}.jsonl'
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for example in training_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"Saved JSONL format to {jsonl_path}")
        
        # Print statistics
        self._print_statistics(training_data)
        
        return training_data
    
    def _print_statistics(self, training_data: list[dict]):
        """Print statistics about generated training data"""
        print("\n" + "="*50)
        print("Training Data Statistics")
        print("="*50)
        
        # Count by type
        type_counts = {}
        for example in training_data:
            if 'metadata' in example:
                t = example['metadata'].get('type', 'unknown')
                type_counts[t] = type_counts.get(t, 0) + 1
        
        print("\nBy component type:")
        for t, count in sorted(type_counts.items()):
            print(f"  {t}: {count}")
        
        # Content length statistics
        content_lengths = [
            len(ex['messages'][2]['content'])
            for ex in training_data
        ]
        
        if content_lengths:
            print(f"\nHTML content length:")
            print(f"  Min: {min(content_lengths)} chars")
            print(f"  Max: {max(content_lengths)} chars")
            print(f"  Avg: {sum(content_lengths) // len(content_lengths)} chars")
        
        print("="*50 + "\n")


# ============================================
# CLI
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate fine-tuning training data from HTML files'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=Path('data/raw_html'),
        help='Input directory containing HTML files'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('data/training'),
        help='Output directory for training data'
    )
    
    parser.add_argument(
        '--provider', '-p',
        type=str,
        choices=['gemini', 'ollama'],
        default=None,
        help='LLM provider (default: from .env)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Model name (default: from .env)'
    )
    
    parser.add_argument(
        '--language', '-l',
        type=str,
        choices=['ko', 'en'],
        default=None,
        help='Language for generated instructions (default: from .env)'
    )
    
    parser.add_argument(
        '--num-instructions', '-n',
        type=int,
        default=None,
        help='Number of instructions per component (default: from .env)'
    )
    
    parser.add_argument(
        '--categories', '-c',
        type=str,
        default=None,
        help='Comma-separated list of categories to extract (page,section,component)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=None,
        help='Batch size for LLM inference (default: 4)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test LLM connection and exit'
    )
    
    args = parser.parse_args()
    
    # Get configuration from environment with CLI overrides
    language = args.language or os.getenv('INSTRUCTION_LANGUAGE', 'ko')
    num_instructions = args.num_instructions or int(os.getenv('NUM_INSTRUCTIONS_PER_COMPONENT', '3'))
    min_html = int(os.getenv('MIN_HTML_LENGTH', '50'))
    max_html = int(os.getenv('MAX_HTML_LENGTH', '10000'))
    batch_size = args.batch_size or int(os.getenv('BATCH_SIZE', '4'))
    
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(',')]
    elif os.getenv('EXTRACT_CATEGORIES'):
        categories = [c.strip() for c in os.getenv('EXTRACT_CATEGORIES').split(',')]
    
    # Initialize LLM client
    provider = args.provider or os.getenv('LLM_PROVIDER', 'gemini')
    
    print(f"Using LLM provider: {provider}")
    
    try:
        if provider == 'gemini':
            from src.utils.llm_client import GeminiClient
            llm_client = GeminiClient(model=args.model)
        else:
            from src.utils.llm_client import OllamaClient
            llm_client = OllamaClient(model=args.model)
    except Exception as e:
        print(f"Error initializing LLM client: {e}")
        sys.exit(1)
    
    # Test mode
    if args.test:
        print("Testing LLM connection...")
        if llm_client.is_available():
            print("✓ LLM is available")
            response = llm_client.generate("Say 'Hello'")
            print(f"✓ Test response: {response[:100]}")
        else:
            print("✗ LLM is not available")
            if provider == 'ollama':
                print("  Make sure Ollama is running and the model is pulled")
            sys.exit(1)
        return
    
    # Check input directory
    if not args.input.exists():
        print(f"Error: Input directory does not exist: {args.input}")
        print(f"Please create the directory and add HTML files to process.")
        sys.exit(1)
    
    # Run pipeline
    pipeline = TrainingDataPipeline(
        llm_client=llm_client,
        language=language,
        num_instructions=num_instructions,
        min_html_length=min_html,
        max_html_length=max_html,
        extract_categories=categories,
        batch_size=batch_size
    )
    
    pipeline.process_directory(args.input, args.output)


if __name__ == '__main__':
    main()
