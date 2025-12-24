import dashscope
from typing import List, Dict, Any
from config.settings import Settings

class QwenLLMClient:
    def __init__(self):
        self.api_key = Settings.QWEN_API_KEY
        self.model = Settings.LLM_MODEL
        dashscope.api_key = self.api_key

    def generate_response(self, prompt: str, context: List[str] = None) -> str:
        """
        Generate response using Qwen3-Max model
        """
        if context:
            # Combine context with the original prompt
            context_str = "\n".join(context)
            full_prompt = f"请根据以下上下文信息回答问题：\n{context_str}\n\n问题：{prompt}"
        else:
            full_prompt = prompt

        try:
            response = dashscope.Generation.call(
                model=self.model,
                prompt=full_prompt,
                max_tokens=1024,
                temperature=0.7
            )
            
            if response.status_code == 200:
                return response.output.text
            else:
                raise Exception(f"API Error: {response.message}")
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return "抱歉，我在处理您的请求时遇到了问题。"

    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """
        Chat completion for conversation-style interaction
        """
        try:
            response = dashscope.ChatCompletion.call(
                model=self.model,
                messages=messages
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                raise Exception(f"API Error: {response.message}")
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return "抱歉，我在处理您的请求时遇到了问题。"