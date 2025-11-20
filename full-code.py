import litellm
from litellm import CustomLLM
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import requests
from typing import Optional, Dict, Any
import asyncio
import json

# ============================================================================
# CUSTOM FASTAPI LLM WRAPPER
# ============================================================================

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
        
        try:
            # Make request to your FastAPI service
            response = requests.post(
                f"{self.api_base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
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
        except requests.exceptions.RequestException as e:
            print(f"Error calling FastAPI LLM: {e}")
            raise

# ============================================================================
# REGISTER CUSTOM LLM PROVIDER
# ============================================================================

# Update these with your actual FastAPI service details
FASTAPI_BASE_URL = "http://your-fastapi-service.com"  # Replace with your actual URL
FASTAPI_API_KEY = "your-api-key"  # Replace with your actual API key
FASTAPI_MODEL_NAME = "your-model-name"  # Replace with your model name

# Register your custom LLM provider
litellm.custom_provider_map = [
    {
        "provider": "fastapi_custom",
        "custom_handler": FastAPILLMWrapper(
            api_base_url=FASTAPI_BASE_URL,
            api_key=FASTAPI_API_KEY
        )
    }
]

# ============================================================================
# CUSTOM TOOLS FOR AGENTS
# ============================================================================

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyzes the sentiment of the provided text.
    
    Args:
        text (str): The text to analyze for sentiment.
    
    Returns:
        dict: A dictionary containing sentiment analysis results with keys:
              - sentiment: The overall sentiment (positive, negative, neutral)
              - confidence: Confidence score (0-1)
              - emotions: List of detected emotions
    """
    # This is a placeholder - your LLM agent will actually do the analysis
    # This function just formats the expected output structure
    return {
        "status": "success",
        "message": f"Sentiment analysis completed for text: {text[:50]}..."
    }

def classify_intent(text: str) -> Dict[str, Any]:
    """
    Classifies the intent of the provided text.
    
    Args:
        text (str): The text to classify for intent.
    
    Returns:
        dict: A dictionary containing intent classification results with keys:
              - intent: The classified intent category
              - confidence: Confidence score (0-1)
              - sub_intents: List of detected sub-intents if any
    """
    # This is a placeholder - your LLM agent will actually do the classification
    # This function just formats the expected output structure
    return {
        "status": "success",
        "message": f"Intent classification completed for text: {text[:50]}..."
    }

# ============================================================================
# AGENT 1: SENTIMENT ANALYSIS AGENT
# ============================================================================

sentiment_agent = LlmAgent(
    model=LiteLlm(
        model=f"fastapi_custom/{FASTAPI_MODEL_NAME}",
        api_base=FASTAPI_BASE_URL
    ),
    name="sentiment_analyzer",
    description="Agent specialized in analyzing sentiment of text. Detects positive, negative, or neutral sentiment with confidence scores and emotional analysis.",
    instruction="""You are an expert sentiment analysis agent. Your task is to analyze the sentiment of the provided text.

For each analysis, you must provide:
1. Overall sentiment: Classify as 'positive', 'negative', or 'neutral'
2. Confidence score: A number between 0 and 1 indicating your confidence
3. Emotions detected: List specific emotions (e.g., joy, anger, sadness, fear, surprise, disgust)
4. Key phrases: Highlight phrases that influenced your sentiment determination
5. Sentiment intensity: Rate on a scale of 1-10

Format your response as JSON with the following structure:
{
  "sentiment": "positive/negative/neutral",
  "confidence": 0.95,
  "emotions": ["joy", "excitement"],
  "key_phrases": ["great experience", "loved it"],
  "intensity": 8,
  "reasoning": "Brief explanation of your analysis"
}

Be thorough, objective, and consider context, sarcasm, and nuanced language.""",
    tools=[analyze_sentiment],
    output_key="sentiment_result"
)

# ============================================================================
# AGENT 2: INTENT CLASSIFICATION AGENT
# ============================================================================

intent_agent = LlmAgent(
    model=LiteLlm(
        model=f"fastapi_custom/{FASTAPI_MODEL_NAME}",
        api_base=FASTAPI_BASE_URL
    ),
    name="intent_classifier",
    description="Agent specialized in classifying user intent from text. Identifies what the user wants to accomplish or communicate.",
    instruction="""You are an expert intent classification agent. Your task is to classify the intent behind the provided text.

Common intent categories include:
- question_seeking_information
- complaint_issue_reporting
- request_action
- feedback_opinion
- greeting_social
- booking_reservation
- cancellation_refund
- technical_support
- price_inquiry
- product_inquiry
- general_conversation

For each classification, you must provide:
1. Primary intent: The main intent category
2. Confidence score: A number between 0 and 1
3. Sub-intents: Any secondary or supporting intents
4. Entities: Extract key entities (dates, products, names, etc.)
5. Urgency level: Rate urgency from 1-5
6. Actionable items: What actions are implied or requested

Format your response as JSON with the following structure:
{
  "primary_intent": "request_action",
  "confidence": 0.92,
  "sub_intents": ["price_inquiry"],
  "entities": {
    "product": "laptop",
    "price_range": "under $1000"
  },
  "urgency": 3,
  "actionable_items": ["provide product recommendations", "share pricing information"],
  "reasoning": "Brief explanation of your classification"
}

Consider context, implicit needs, and multiple possible intents.""",
    tools=[classify_intent],
    output_key="intent_result"
)

# ============================================================================
# SESSION AND RUNNER SETUP
# ============================================================================

APP_NAME = "text_analysis_app"
USER_ID = "user_001"
SESSION_ID = "session_001"

async def setup_session_and_runner(agent: LlmAgent):
    """Setup session and runner for an agent"""
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=f"{SESSION_ID}_{agent.name}"
    )
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    return session, runner

# ============================================================================
# AGENT EXECUTION FUNCTIONS
# ============================================================================

async def run_sentiment_analysis(text: str) -> Dict[str, Any]:
    """
    Run sentiment analysis on the provided text.
    
    Args:
        text (str): Text to analyze
    
    Returns:
        dict: Sentiment analysis results
    """
    print(f"\n{'='*70}")
    print(f"SENTIMENT ANALYSIS")
    print(f"{'='*70}")
    print(f"Input Text: {text}\n")
    
    content = types.Content(
        role='user',
        parts=[types.Part(text=f"Analyze the sentiment of this text: {text}")]
    )
    
    session, runner = await setup_session_and_runner(sentiment_agent)
    
    events = runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=content
    )
    
    final_response = None
    async for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response:")
            print(final_response)
            print(f"\n{'='*70}\n")
    
    # Try to parse JSON response
    try:
        result = json.loads(final_response)
        return result
    except json.JSONDecodeError:
        return {"raw_response": final_response}

async def run_intent_classification(text: str) -> Dict[str, Any]:
    """
    Run intent classification on the provided text.
    
    Args:
        text (str): Text to classify
    
    Returns:
        dict: Intent classification results
    """
    print(f"\n{'='*70}")
    print(f"INTENT CLASSIFICATION")
    print(f"{'='*70}")
    print(f"Input Text: {text}\n")
    
    content = types.Content(
        role='user',
        parts=[types.Part(text=f"Classify the intent of this text: {text}")]
    )
    
    session, runner = await setup_session_and_runner(intent_agent)
    
    events = runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=content
    )
    
    final_response = None
    async for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response:")
            print(final_response)
            print(f"\n{'='*70}\n")
    
    # Try to parse JSON response
    try:
        result = json.loads(final_response)
        return result
    except json.JSONDecodeError:
        return {"raw_response": final_response}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main function to demonstrate both agents"""
    
    # Test cases
    test_texts = [
        "I absolutely loved this product! The quality exceeded my expectations and the customer service was amazing.",
        "I need to cancel my subscription and get a refund immediately.",
        "How much does the premium plan cost?",
        "This is the worst experience I've ever had. The product broke after just one day!",
        "Can you help me reset my password? I can't access my account."
    ]
    
    print("\n" + "="*70)
    print("TESTING SENTIMENT ANALYSIS AND INTENT CLASSIFICATION AGENTS")
    print("="*70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n\nTest Case {i}/{len(test_texts)}")
        print("-" * 70)
        
        # Run sentiment analysis
        sentiment_result = await run_sentiment_analysis(text)
        
        # Run intent classification
        intent_result = await run_intent_classification(text)
        
        # Summary
        print("\n" + "="*70)
        print(f"SUMMARY FOR TEST CASE {i}")
        print("="*70)
        print(f"Text: {text}")
        print(f"\nSentiment: {sentiment_result}")
        print(f"\nIntent: {intent_result}")
        print("="*70)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
