"""
LLM Client module supporting multiple providers (Gemini, Ollama)
"""
import os
import math
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from tqdm import tqdm

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response from the LLM"""
        pass
    
    @abstractmethod
    def generate_batch(self, prompts: list[str], system_prompt: Optional[str] = None, batch_size: int = 4) -> list[str]:
        """Generate responses for multiple prompts in batches (faster than calling generate multiple times)"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available"""
        pass


class GeminiClient(LLMClient):
    """Google Gemini API client"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self._model = None
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiClient")
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the Gemini client"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
        except ImportError:
            raise ImportError("google-generativeai package is required. Install with: pip install google-generativeai")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using Gemini API"""
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = self._model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}")
    
    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
        num_workers: int = 4,
        batch_size: Optional[int] = None  # Deprecated, use num_workers
    ) -> list[str]:
        """
        Generate responses for multiple prompts in PARALLEL.
        
        Args:
            prompts: List of prompts to generate responses for
            system_prompt: Optional system prompt to include in each request
            num_workers: Number of concurrent requests (threads) to run in parallel
            batch_size: Deprecated, use num_workers instead
        
        Returns:
            List of generated responses in the same order as input prompts
        """
        # Handle deprecated batch_size parameter
        workers = batch_size if batch_size is not None else num_workers
        if workers < 1:
            raise ValueError("num_workers must be >= 1")

        total_prompts = len(prompts)
        if total_prompts == 0:
            return []
        
        # Pre-allocate results list to maintain order
        results = [""] * total_prompts
        
        def process_single(idx: int, prompt: str) -> tuple[int, str]:
            """Process a single prompt and return (index, result)"""
            try:
                result = self.generate(prompt, system_prompt)
                return (idx, result)
            except Exception as e:
                tqdm.write(f"Warning: Gemini generation failed for prompt {idx}: {e}")
                return (idx, "")
        
        # Use ThreadPoolExecutor for true parallel execution
        with tqdm(total=total_prompts, desc=f"Gemini parallel inference (workers={workers})", unit="prompt") as pbar:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(process_single, idx, prompt): idx 
                    for idx, prompt in enumerate(prompts)
                }
                
                # Collect results as they complete
                for future in as_completed(futures):
                    idx, result = future.result()
                    results[idx] = result
                    pbar.update(1)

        return results
    
    def is_available(self) -> bool:
        """Check if Gemini API is available"""
        try:
            # Simple test generation
            response = self._model.generate_content("Say 'OK'")
            return bool(response.text)
        except Exception:
            return False


class OllamaClient(LLMClient):
    """Ollama local LLM client"""
    
    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model_name = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self._client = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the Ollama client"""
        try:
            import ollama
            self._client = ollama.Client(host=self.host)
        except ImportError:
            raise ImportError("ollama package is required. Install with: pip install ollama")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using Ollama"""
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = self._client.chat(
                model=self.model_name,
                messages=messages
            )
            
            return response["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
        num_workers: int = 4,
        batch_size: Optional[int] = None  # Deprecated, use num_workers
    ) -> list[str]:
        """
        Generate responses for multiple prompts in PARALLEL.
        
        Args:
            prompts: List of prompts to generate responses for
            system_prompt: Optional system prompt to include in each request
            num_workers: Number of concurrent requests (threads) to run in parallel
                        NOTE: Ollama server must support parallel requests.
                        Set OLLAMA_NUM_PARALLEL env var on server side to match.
            batch_size: Deprecated, use num_workers instead
        
        Returns:
            List of generated responses in the same order as input prompts
        """
        # Handle deprecated batch_size parameter
        workers = batch_size if batch_size is not None else num_workers
        if workers < 1:
            raise ValueError("num_workers must be >= 1")

        total_prompts = len(prompts)
        if total_prompts == 0:
            return []
        
        # Pre-allocate results list to maintain order
        results = [""] * total_prompts
        
        def process_single(idx: int, prompt: str) -> tuple[int, str]:
            """Process a single prompt and return (index, result)"""
            try:
                result = self.generate(prompt, system_prompt)
                return (idx, result)
            except Exception as e:
                tqdm.write(f"Warning: Ollama generation failed for prompt {idx}: {e}")
                return (idx, "")
        
        # Use ThreadPoolExecutor for true parallel execution
        with tqdm(total=total_prompts, desc=f"Ollama parallel inference (workers={workers})", unit="prompt") as pbar:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(process_single, idx, prompt): idx 
                    for idx, prompt in enumerate(prompts)
                }
                
                # Collect results as they complete
                for future in as_completed(futures):
                    idx, result = future.result()
                    results[idx] = result
                    pbar.update(1)
        
        return results
    
    def is_available(self) -> bool:
        """Check if Ollama server is available and model exists"""
        try:
            # Check server is running
            models = self._client.list()
            model_names = [m["name"] for m in models.get("models", [])]
            
            # Check if model exists (with or without :latest tag)
            base_model = self.model_name.split(":")[0]
            for name in model_names:
                if name.startswith(base_model):
                    return True
            
            print(f"Warning: Model '{self.model_name}' not found. Available models: {model_names}")
            return False
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            return False
    
    def pull_model(self) -> bool:
        """Pull the model if not already available"""
        try:
            print(f"Pulling model '{self.model_name}'... This may take a while.")
            self._client.pull(self.model_name)
            return True
        except Exception as e:
            print(f"Failed to pull model: {e}")
            return False


def get_llm_client(provider: Optional[str] = None, **kwargs) -> LLMClient:
    """
    Factory function to get the appropriate LLM client
    
    Args:
        provider: "gemini" or "ollama". If None, reads from LLM_PROVIDER env var
        **kwargs: Additional arguments passed to the client constructor
    
    Returns:
        LLMClient instance
    """
    provider = provider or os.getenv("LLM_PROVIDER", "gemini")
    provider = provider.lower()
    
    if provider == "gemini":
        return GeminiClient(**kwargs)
    elif provider == "ollama":
        return OllamaClient(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Supported: gemini, ollama")


# Convenience function for quick testing
def test_llm_client():
    """Test the configured LLM client"""
    provider = os.getenv("LLM_PROVIDER", "gemini")
    print(f"Testing LLM provider: {provider}")
    
    try:
        client = get_llm_client()
        
        if client.is_available():
            print("✓ LLM service is available")
            
            response = client.generate("Say 'Hello, World!' in Korean")
            print(f"✓ Test response: {response[:100]}...")
            return True
        else:
            print("✗ LLM service is not available")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    test_llm_client()
