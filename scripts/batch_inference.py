#!/usr/bin/env python
"""
Batch Inference Utility for Ollama

This script provides utilities for batch processing with Ollama to speed up inference:
1. Batch generate instructions for multiple components
2. Batch generate responses for custom prompts
3. Parallel processing support (for compatible models)

Usage:
    # Generate instructions for extracted components
    python scripts/batch_inference.py --input data/extracted_components.json --batch-size 8
    
    # Generate responses for custom prompts
    python scripts/batch_inference.py --prompts data/my_prompts.jsonl --batch-size 4 --output results.jsonl
    
    # Test batch performance
    python scripts/batch_inference.py --benchmark --num-items 100 --batch-size 4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.llm_client import get_llm_client, OllamaClient

# Load environment variables
load_dotenv()


class BatchInferenceEngine:
    """Engine for batch inference with progress tracking"""
    
    def __init__(self, llm_client, batch_size: int = 4, show_progress: bool = True):
        self.llm_client = llm_client
        self.batch_size = batch_size
        self.show_progress = show_progress
    
    def process_components(
        self,
        components: list[dict],
        system_prompt: Optional[str] = None,
        component_prompt_template: Optional[str] = None
    ) -> list[dict]:
        """
        Process components (with HTML content) and generate responses
        
        Args:
            components: List of dicts with 'content' and optionally 'type' keys
            system_prompt: System prompt to use
            component_prompt_template: Template for generating prompts from components
        
        Returns:
            List of results with original data + 'response' field
        """
        prompts = []
        
        if component_prompt_template:
            # Custom prompt template
            for component in components:
                prompt = component_prompt_template.format(
                    content=component.get('content', ''),
                    type=component.get('type', 'component')
                )
                prompts.append(prompt)
        else:
            # Use content as prompt directly
            prompts = [c.get('content', '') for c in components]
        
        # Generate responses using batch
        if hasattr(self.llm_client, 'generate_batch'):
            responses = self.llm_client.generate_batch(
                prompts,
                system_prompt=system_prompt,
                batch_size=self.batch_size
            )
        else:
            # Fallback to sequential processing
            if self.show_progress:
                print("LLM doesn't support batch processing, using sequential...")
            responses = []
            for prompt in tqdm(prompts, desc="Generating responses"):
                response = self.llm_client.generate(prompt, system_prompt)
                responses.append(response)
        
        # Combine results
        results = []
        for component, response in zip(components, responses):
            result = {**component, 'response': response}
            results.append(result)
        
        return results
    
    def process_prompts(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None
    ) -> list[dict]:
        """
        Process a list of prompts and generate responses
        
        Args:
            prompts: List of prompts
            system_prompt: System prompt to use
        
        Returns:
            List of dicts with 'prompt' and 'response' fields
        """
        if hasattr(self.llm_client, 'generate_batch'):
            responses = self.llm_client.generate_batch(
                prompts,
                system_prompt=system_prompt,
                batch_size=self.batch_size
            )
        else:
            if self.show_progress:
                print("LLM doesn't support batch processing, using sequential...")
            responses = []
            for prompt in tqdm(prompts, desc="Generating responses"):
                response = self.llm_client.generate(prompt, system_prompt)
                responses.append(response)
        
        # Combine results
        results = []
        for prompt, response in zip(prompts, responses):
            results.append({
                'prompt': prompt,
                'response': response
            })
        
        return results
    
    def benchmark(
        self,
        num_items: int = 100,
        prompt_length: int = 100,
        system_prompt: Optional[str] = None
    ) -> dict:
        """
        Benchmark batch processing performance
        
        Args:
            num_items: Number of items to process
            prompt_length: Length of each test prompt
            system_prompt: System prompt to use
        
        Returns:
            Dict with timing and performance metrics
        """
        # Generate test prompts
        test_prompt = "What is the meaning of life?" * (prompt_length // 33)
        prompts = [test_prompt] * num_items
        
        print(f"\nBenchmarking with {num_items} prompts (batch size: {self.batch_size})...")
        print(f"Prompt template length: {len(test_prompt)} chars")
        
        start_time = time.time()
        
        if hasattr(self.llm_client, 'generate_batch'):
            responses = self.llm_client.generate_batch(
                prompts,
                system_prompt=system_prompt,
                batch_size=self.batch_size
            )
        else:
            responses = []
            for prompt in tqdm(prompts):
                response = self.llm_client.generate(prompt, system_prompt)
                responses.append(response)
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            'num_items': num_items,
            'batch_size': self.batch_size,
            'total_time_seconds': elapsed,
            'time_per_item_seconds': elapsed / num_items,
            'items_per_second': num_items / elapsed,
            'prompt_length': len(test_prompt),
            'avg_response_length': sum(len(r) for r in responses) / len(responses) if responses else 0
        }
        
        return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Batch inference utility for faster LLM processing'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        help='Input JSON/JSONL file with components or prompts'
    )
    
    parser.add_argument(
        '--prompts',
        type=Path,
        help='Input file with prompts (one per line or JSON array)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output file for results (default: input_batch_results.json)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=int(os.getenv('BATCH_SIZE', '4')),
        help='Number of items to process per batch (default: 4)'
    )
    
    parser.add_argument(
        '--system-prompt', '-s',
        type=str,
        default=None,
        help='System prompt to include in requests'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Model name (default: from .env)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='Ollama host (default: from .env or localhost:11434)'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark test instead of processing files'
    )
    
    parser.add_argument(
        '--num-items',
        type=int,
        default=100,
        help='Number of items for benchmark (default: 100)'
    )
    
    parser.add_argument(
        '--prompt-length',
        type=int,
        default=100,
        help='Approximate prompt length for benchmark (default: 100 chars)'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'jsonl'],
        default='json',
        help='Output format (default: json)'
    )
    
    args = parser.parse_args()
    
    # Initialize LLM client
    print("Initializing LLM client...")
    try:
        llm_client = get_llm_client(
            provider='ollama',
            model=args.model,
            host=args.host
        )
        
        if not llm_client.is_available():
            print("✗ LLM not available")
            sys.exit(1)
        
        print("✓ LLM is available")
    except Exception as e:
        print(f"Error initializing LLM client: {e}")
        sys.exit(1)
    
    # Create inference engine
    engine = BatchInferenceEngine(
        llm_client,
        batch_size=args.batch_size,
        show_progress=True
    )
    
    # Benchmark mode
    if args.benchmark:
        metrics = engine.benchmark(
            num_items=args.num_items,
            prompt_length=args.prompt_length,
            system_prompt=args.system_prompt
        )
        
        print("\n" + "="*60)
        print("Benchmark Results")
        print("="*60)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("="*60)
        
        return
    
    # Process mode
    if args.prompts:
        print(f"\nLoading prompts from {args.prompts}...")
        
        # Load prompts
        prompts = []
        try:
            with open(args.prompts, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip().startswith('['):
                    # JSON array format
                    prompts = json.loads(content)
                else:
                    # JSONL or line-separated format
                    for line in content.strip().split('\n'):
                        if line.strip():
                            prompts.append(line.strip())
        except Exception as e:
            print(f"Error loading prompts: {e}")
            sys.exit(1)
        
        print(f"Loaded {len(prompts)} prompts")
        
        # Process
        print(f"\nProcessing {len(prompts)} prompts with batch size {args.batch_size}...")
        results = engine.process_prompts(
            prompts,
            system_prompt=args.system_prompt
        )
    
    elif args.input:
        print(f"\nLoading components from {args.input}...")
        
        # Load components
        components = []
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip().startswith('['):
                    # JSON array format
                    components = json.loads(content)
                else:
                    # JSONL format
                    for line in content.strip().split('\n'):
                        if line.strip():
                            components.append(json.loads(line))
        except Exception as e:
            print(f"Error loading components: {e}")
            sys.exit(1)
        
        print(f"Loaded {len(components)} components")
        
        # Process
        print(f"\nProcessing {len(components)} components with batch size {args.batch_size}...")
        results = engine.process_components(
            components,
            system_prompt=args.system_prompt
        )
    
    else:
        print("Please provide either --input, --prompts, or --benchmark")
        parser.print_help()
        sys.exit(1)
    
    # Determine output path
    if args.output is None:
        if args.input:
            output_path = args.input.parent / f"{args.input.stem}_batch_results.{args.format}"
        else:
            output_path = args.prompts.parent / f"{args.prompts.stem}_batch_results.{args.format}"
    else:
        output_path = args.output
    
    # Save results
    print(f"\nSaving {len(results)} results to {output_path}...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    print(f"Total items processed: {len(results)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output format: {args.format}")
    print("="*60)


if __name__ == '__main__':
    main()
