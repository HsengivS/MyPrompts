# Lite LLM

import litellm
from litellm import CustomLLM
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
import requests
from typing import Optional, Union

class FastAPILLMWrapper(CustomLLM):
    def __init__(self, api_base_url: str, api_key: Optional[str] = None):
        super().__init__()
        self.api_base_url = api_base_url
        self.api_key = api_key
    
    def completion(
        self,
        model: str,
        messages: list,
        api_base: Optional[str] = None,
        custom_llm_provider: str = "custom",
        **kwargs
    ) -> litellm.ModelResponse:
        """
        Handle completion requests to your FastAPI endpoint
        """
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Prepare request payload for your FastAPI endpoint
        payload = {
            "messages": messages,
            "model": model,
            **kwargs  # Include temperature, max_tokens, etc.
        }
        
        # Make request to your FastAPI service
        response = requests.post(
            f"{self.api_base_url}/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        # Convert response to LiteLLM format
        data = response.json()
        
        # Transform to litellm.ModelResponse format
        return litellm.ModelResponse(
            id=data.get("id", "custom-id"),
            choices=[{
                "message": {
                    "role": "assistant",
                    "content": data["choices"][0]["message"]["content"]
                },
                "finish_reason": data["choices"][0].get("finish_reason", "stop"),
                "index": 0
            }],
            model=model,
            usage={
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": data.get("usage", {}).get("total_tokens", 0)
            }
        )

# Register your custom LLM provider
litellm.custom_provider_map = [
    {
        "provider": "fastapi_custom",
        "custom_handler": FastAPILLMWrapper(
            api_base_url="http://your-fastapi-service.com",
            api_key="your-api-key"
        )
    }
]

# Use with Google ADK
agent = LlmAgent(
    model=LiteLlm(
        model="fastapi_custom/your-model-name",
        api_base="http://your-fastapi-service.com"
    ),
    name="custom_fastapi_agent",
    description="Agent using custom FastAPI-hosted LLM",
    instruction="You are a helpful assistant.",
    tools=[your_tools]
)


# ----------------------------------------------------------------------------------------------------------------------

# Direct LiteLLM Approach

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# Configure LiteLLM to use OpenAI-compatible endpoint
agent = LlmAgent(
    model=LiteLlm(
        model="openai/your-model-name",
        api_base="http://your-fastapi-service.com/v1",
        api_key="your-api-key",
        custom_llm_provider="openai"  # If your FastAPI follows OpenAI format
    ),
    name="fastapi_agent",
    instruction="You are a helpful assistant.",
    tools=[your_tools]
)

# ----------------------------------------------------------------------------------------------------------------------

# BeeAI Framework

import os
import requests
from typing import Optional, AsyncIterator
from beeai.llms.base import ChatModel
from beeai.agents import RequirementAgent
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

class FastAPILLM(ChatModel):
    """Custom LLM wrapper for FastAPI-hosted models"""
    
    def __init__(
        self,
        api_base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_base_url = api_base_url
        self.model_name = model_name
        self.api_key = api_key
    
    async def generate(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Generate completion from FastAPI endpoint"""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "messages": messages,
            "model": self.model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        response = requests.post(
            f"{self.api_base_url}/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def stream(
        self,
        messages: list,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream completion from FastAPI endpoint"""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "messages": messages,
            "model": self.model_name,
            "stream": True,
            **kwargs
        }
        
        response = requests.post(
            f"{self.api_base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                # Parse SSE format
                if line.startswith(b"data: "):
                    data = line[6:].decode('utf-8')
                    if data != "[DONE]":
                        import json
                        chunk = json.loads(data)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content

# Create A2A server with custom LLM
def main():
    # Initialize custom LLM
    llm = FastAPILLM(
        api_base_url="http://your-fastapi-service.com",
        model_name="your-model-name",
        api_key=os.environ.get("FASTAPI_API_KEY")
    )
    
    # Create agent
    agent = RequirementAgent(
        llm=llm,
        tools=[your_tools],
        memory=your_memory
    )
    
    # Setup A2A server
    agent_card = AgentCard(
        name="FastAPI Agent",
        description="Agent using custom FastAPI-hosted LLM",
        url="http://localhost:9999/",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="custom_skill",
                name="Custom FastAPI Skill",
                description="Handles requests using FastAPI-hosted LLM"
            )
        ]
    )
    
    # Create and run server
    from a2a.server.config import A2AServerConfig
    from a2a.server import A2AServer
    
    server = A2AServer(
        config=A2AServerConfig(port=9999),
        agent_card=agent_card
    ).register(agent).serve()

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------------------------------------------------

# A2A sdk directly

from a2a.client import A2AClient
from a2a.server.apps import A2AStarletteApplication
import requests

class FastAPILLMProvider:
    """Standalone FastAPI LLM provider for A2A"""
    
    def __init__(self, api_base_url: str, api_key: str = None):
        self.api_base_url = api_base_url
        self.api_key = api_key
    
    async def process_task(self, task_input: str, **kwargs):
        """Process A2A task using FastAPI LLM"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "messages": [{"role": "user", "content": task_input}],
            **kwargs
        }
        
        response = requests.post(
            f"{self.api_base_url}/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        return response.json()["choices"][0]["message"]["content"]


# ----------------------------------------------------------------------------------------------------------------------

https://github.com/patchy631/ai-engineering-hub/blob/main/agent2agent-demo/notebook.ipynb























