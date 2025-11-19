import requests
from google.generativeai import GenerativeModel

class EnterpriseAdapter:
    def __init__(self, endpoint_url):
        self.endpoint = endpoint_url
        # generative sdk looks for adapter.model_name
        self.model_name = "enterprise-llm"

    def generate_content(self, contents, **kwargs):
        # extract prompt text (works with the SDK content structure)
        prompt_text = None
        try:
            prompt_text = contents[0].parts[0].text
        except Exception:
            # fallback if caller passes plain string
            prompt_text = contents if isinstance(contents, str) else str(contents)

        payload = {
            "prompt": prompt_text,
            "temperature": kwargs.get("temperature", 0.7)
        }
        r = requests.post(self.endpoint, json=payload, timeout=30)
        r.raise_for_status()
        out = r.json()

        # assume your backend returns {"text": "..."}
        text = out.get("text") or out.get("output") or str(out)

        # return Gemini-compatible envelope
        return {
            "candidates": [
                {"content": {"parts": [{"text": text}]}}
            ]
        }

# usage
adapter = EnterpriseAdapter("http://localhost:8000/generate")
model = GenerativeModel(model_name="enterprise-llm", adapter=adapter)

# test
chat = model.start_chat(system_instruction="You are a sentiment classifier.")
resp = chat.send_message("I love this product!")
print("->", resp.text)
