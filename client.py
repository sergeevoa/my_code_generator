import requests
import json

class OllamaClient:
    """
    Клиент Ollama, работающий локально через Ollama API.
    """

    def __init__(self, model, host):
        self.model = model
        self.api_url = f"{host}/api/generate"

    def generate(self, prompt, max_tokens=200, temperature=0.2):
    
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        response = requests.post(self.api_url, json=payload)

        try:
            response.raise_for_status() # Выбросит ошибку при HTTP response != 200
            
            data = response.json()

            # Ollama API может возвращать текст в разных полях
            if isinstance(data, dict):
                return (data.get("text") or "").strip() \
                    or (data.get("content") or "").strip() \
                    or data
            
            # Если ответ не является словарем, возможно Ollama вернула только строку ответа
            return data
        
        except requests.exceptions.JSONDecodeError:
            raise ValueError(f"Invalid JSON from Ollama: {response.text}")
        
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error calling Ollama: {e}")

    def chat(self, messages, max_tokens=200, temperature=0.2):
        """
        Эмуляция диалогового режима.
        messages - список словарей: {"role": "user"|"assistant", "content": str}
        Список messages конвертируется в одну строку (prompt), которую понимают модели Ollama. 
        """

        prompt = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        prompt += "Assistant: "  # Подсказка для модели, что она - ассистент и должна ответить

        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)