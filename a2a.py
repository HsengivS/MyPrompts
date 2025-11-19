#############################################################
# agent2agent_enterprise_llm_multi_agent.py
#############################################################

import requests
from python_a2a import AgentRouter


#############################################################
# 1. Enterprise Hosted LLM Wrapper
#############################################################

class EnterpriseLLM:
    """
    Wrapper for your enterprise-hosted LLM.
    python_a2a requires only a .generate(prompt) method.
    """

    def __init__(self, endpoint_url: str, api_key: str = None):
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def generate(self, prompt: str, **kwargs) -> str:
        payload = {
            "prompt": prompt,
            **kwargs
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(self.endpoint_url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()

        return (
            data.get("response")
            or data.get("text")
            or data.get("output")
            or str(data)
        )


#############################################################
# 2. Setup Router + LLM Backend
#############################################################

llm = EnterpriseLLM(
    endpoint_url="https://your-enterprise-llm.company.com/v1/generate",
    api_key="YOUR_API_KEY_HERE"
)

router = AgentRouter(llm_backend=llm)


#############################################################
# 3. Independent Agents
#############################################################

# -----------------------------------------------------------
# A. SENTIMENT AGENT (Independent)
# -----------------------------------------------------------
@router.agent
def sentiment_agent(text: str):
    """
    Returns sentiment only (Positive / Neutral / Negative)
    """
    prompt = f"""
Classify the sentiment of the following text as: Positive, Neutral, or Negative.

Respond ONLY with a JSON object:
{{
  "sentiment": "Positive | Neutral | Negative",
  "reason": "short reason"
}}

Text:
{text}
"""
    output = llm.generate(prompt)
    return f"[Sentiment] {output}"


# -----------------------------------------------------------
# B. NETWORK CLASSIFICATION AGENT (Independent)
# -----------------------------------------------------------
@router.agent
def network_classifier_agent(text: str):
    """
    Classify whether the text is about a network issue or non-network issue.
    """
    prompt = f"""
You are a classifier for telecom customer messages.

Classify the text as:
1. "network_issue"
2. "non_network_issue"

Respond with JSON only:
{{
  "classification": "network_issue | non_network_issue",
  "reason": "short explanation"
}}

Text:
{text}
"""
    output = llm.generate(prompt)
    return f"[NetworkClassification] {output}"


# -----------------------------------------------------------
# C. INTENT AGENT (Independent)
# -----------------------------------------------------------
@router.agent
def intent_agent(text: str):
    """
    Return intent. If it's network-related → choose a network intent.
    If it's not network-related → choose a non-network intent.
    """

    prompt = f"""
You are an expert intent classifier for telecom customer messages.

First determine if the text is a network issue OR not.
Then classify the intent accordingly.

NETWORK ISSUE INTENTS:
- slow_internet
- no_internet
- call_drop
- low_signal
- latency_issue
- packet_loss
- 4g/5g_not_working
- intermittent_connection
- speed_fluctuation

NON-NETWORK ISSUE INTENTS:
- billing_query
- plan_upgrade
- recharge_issue
- sim_replacement
- doma_validation
- customer_verification
- general_inquiry
- complaint_general

Respond in clean JSON only:
{{
  "network_related": true/false,
  "intent": "one of the above intents",
  "reason": "short justification"
}}

Text:
{text}
"""
    output = llm.generate(prompt)
    return f"[Intent] {output}"


#############################################################
# 4. Multi-Agent Parallel Workflow
#############################################################

if __name__ == "__main__":
    print("\n=== STARTING MULTI-AGENT WORKFLOW ===\n")

    user_message = "My internet keeps disconnecting every few minutes and the signal is very weak."

    # Run sentiment, network classification, and intent INDEPENDENTLY
    output = router.run(
        start=None,  # no start agent, direct input
        message=user_message,
        next=[
            "sentiment_agent",
            "network_classifier_agent",
            "intent_agent"
        ]
    )

    print("\n=== FINAL PARALLEL OUTPUT ===")
    print(output)
