from config import LLMConfig
from typing import List
import os
import logging 
import torch
logger = logging.getLogger(__name__)

try:
    import ollama
except ImportError:
    ollama = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# CMU Gateway configuration
CMU_GATEWAY_BASE_URL = "https://ai-gateway.andrew.cmu.edu"
GATEWAY_API_KEY = "sk-gx6H5IC6DMWM311Ef2Gdyw"

class LLM:
    def __init__(self, config: LLMConfig):
        self.config: LLMConfig = config

    def generate(self, query: str, context: List[str]) -> str:
        return f"{query=}, {context=}"
    

class GPT5Mini(LLM):
    """API-based generator using GPT-5-mini via CMU Gateway."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if OpenAI is None:
            raise ImportError("openai package required for GPT5Mini. Install with: pip install openai")
        
        env_key = os.getenv("CMU_GATEWAY_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_key = env_key if env_key and env_key.startswith('sk-') else GATEWAY_API_KEY
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=CMU_GATEWAY_BASE_URL
        )
        self.model_name = "gpt-5-mini"
    
    def generate(self, query: str, context: List[str]) -> str:
        # Different prompts based on whether retrieval is used
        if not context:
            # No-retrieval mode: just answer the question directly
            messages = [
                {
                    'role': 'system',
                    'content': 'You are an extremely knowledgeable and helpful assistant. Answer questions accurately and concisely.'
                },
                {
                    'role': 'user',
                    'content': f"Question: {query}"
                }
            ]
        else:
            # RAG mode: use only the provided context
            context_text = "\n".join(context)
            messages = [
                {
                    'role': 'system',
                    'content': 'You are an extremely knowledgeable and helpful assistant. Use ONLY the provided context to answer questions exactly and accurately. Do not imagine or make up your own answers. The answers should be brief and to the point and contain all the important keywords. You are being graded for the answers so make sure they are correct, and answer the question asked exactly.'
                },
                {
                    'role': 'user',
                    'content': f"Context:\n{context_text}\n\nQuestion: {query}"
                }
            ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0
        )
        
        return response.choices[0].message.content

class Qwen(LLM):
    """Open-weight generator using Qwen/Qwen3-4B-Instruct-2507 from HuggingFace."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("transformers and torch packages required for Qwen. Install with: pip install transformers torch")
        
        self.torch = torch
        
        self.model_name = "Qwen/Qwen3-4B-Instruct-2507"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading Qwen model {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Qwen model loaded successfully")
    
    def generate(self, query: str, context: List[str]) -> str:
        # Different prompts based on whether retrieval is used
        if not context:
            # No-retrieval mode: just answer the question directly
            messages = [
                {
                    "role": "system",
                    "content": "You are an extremely knowledgeable and helpful assistant. Answer questions accurately, concisely, and crisply. Provide complete answers but avoid unnecessary elaboration. For factoid questions, aim for 5-15 words. For list questions, provide numbered items without extra explanation. For multiple choice, select only the correct option letter. Include all important keywords but keep responses tight and focused."
                },
                {
                    "role": "user",
                    "content": f"Question: {query}"
                }
            ]
        else:
            # RAG mode: use only the provided context
            context_text = "\n".join(context)
            messages = [
                {
                    "role": "system",
                    "content": "You are an extremely knowledgeable and helpful assistant. Use ONLY the provided context to answer questions exactly and accurately. Do not imagine or make up your own answers. Provide concise, crisp answers: factoid questions should be 5-15 words, list questions should be numbered items without extra explanation, multiple choice should be just the option letter. Include all important keywords but avoid verbose explanations or unnecessary details. You are being graded for accuracy and conciseness."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {query}"
                }
            ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with self.torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=256,
                do_sample=False,
                temperature=None  # Remove temperature for deterministic output
            )
        
        # Decode only the generated part (excluding the input)
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

class LLaMa(LLM):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.model_name = 'llama3' 
        self.temperature = 0
        
    def generate(self, query: str, context: List[str]) -> str:
        if ollama is None:
            raise ImportError("ollama package is required for LLaMa generator")
        try:
            messages = [    
                {
                    'role': 'system',
                    'content': 'You are an extremely knowledgeable and helpful assistant. Use ONLY the provided context to answer questions exactly and accurately. Do not imagine or make up your own answers. The answers should be brief and to the point and contain all the important keywords. You are being graded for the answers so make sure they are correct, and answer the question asked exactly.'
                },
                {
                    'role': 'user',
                    'content': f"Context:\n{' '.join(context)}\n\nQuestion: {query}"
                }
            ]
            
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={'temperature': self.temperature},
            )
            
            return response['message']['content']
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I couldn't generate an answer at this time."

