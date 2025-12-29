import logging
import json
import re
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from types import SimpleNamespace

from google.adk.models.lite_llm import LiteLlm
from .memory import MemoryLayer
from .tools import search_vendors_tool
from .config import Config
from litellm import completion

class ADKProcurementAgent:
    def __init__(self):
        self.memory = MemoryLayer()
        self.pending_order = None

        # 1. Define Tools
        def consult_memory_tool(query: str) -> str:
            """
            Useful for finding site-specific rules, banned vendors, or spending limits.
            Input: A search query (e.g., 'rules for Mumbai site').
            Output: A list of relevant facts/rules.
            """
            results = self.memory.search(query)
            if not results:
                return "No specific rules found in memory."
            return "\n".join([f"- {m['fact']}" for m in results])

        def search_vendors_wrapper(product_name: str) -> str:
            """
            Useful for finding 3 vendors for a product with their prices.
            Input: Product name (e.g., 'cement').
            Output: JSON string of vendors.
            """
            return search_vendors_tool(product_name)

        # 2. Initialize Model
        # Using LiteLlm from ADK to leverage our existing Config
        self.model = LiteLlm(model=Config.LLM_MODEL_STRING)

        # 3. Create ADK Agent

        self.agent = Agent(
            name="ProcurementBot",
            model=self.model,
            tools=[consult_memory_tool, search_vendors_wrapper],
            instruction="""
You are an Intelligent Procurement Agent.
Your process:
1. ALWAYS use the `consult_memory_tool` to check for site rules (budget limits, banned vendors).
2. Use `search_vendors_wrapper` to find prices.
3. Compare costs against limits found in memory.
4. If Total Cost > Limit OR Vendor is Banned -> Set status to PAUSE_APPROVAL_NEEDED.

Output format:
You MUST explain your reasoning in the response text, AND conclude with a JSON block:
```json
{
    "selected_vendor": "...",
    "price_per_unit": ...,
    "total_cost": ...,
    "status": "ORDER_PLACED" or "PAUSE_APPROVAL_NEEDED",
    "reasoning": "..."
}
```
"""
        )
        # Initialize Runner
        self.runner = Runner(
            agent=self.agent,
            app_name="ProcurementApp",
            session_service=InMemorySessionService()
        )
        # Create the session immediately since it's in-memory and ephemeral
        self.runner.session_service.create_session_sync(
            app_name="ProcurementApp",
            user_id="default_user",
            session_id="default_session"
        )

    def get_intent(self, user_input: str) -> str:
        """
        Determines if the user wants to STORE_FACT (teach) or PROCUREMENT_REQUEST (buy).
        """
        if self.pending_order:
             return "APPROVAL_REPLY"

        prompt = f"""
        Classify the following user input into one of these categories:
        1. STORE_FACT: The user is defining a rule, limit, preference, or fact about a site.
        2. PROCUREMENT_REQUEST: The user is asking to buy, order, or procure something.
        3. CHAT: General conversation (hello, thanks, etc.).

        Input: "{user_input}"
        
        Return ONLY the Category Name (STORE_FACT, PROCUREMENT_REQUEST, or CHAT).
        """
        try:
            # We can use the agent's model directly for lightweight classification if exposed, 
            # but sticking to litellm.completion for simplicity/directness here.
            response = completion(model=Config.LLM_MODEL_STRING, messages=[{"role": "user", "content": prompt}])
            intent = response.choices[0].message.content.strip().upper()
            if "STORE_FACT" in intent: return "STORE_FACT"
            if "PROCUREMENT" in intent: return "PROCUREMENT_REQUEST"
            return "CHAT"
        except Exception:
            return "CHAT"

    def process_message(self, user_input: str) -> dict:
        intent = self.get_intent(user_input)

        if intent == "APPROVAL_REPLY":
            return self.handle_approval(user_input)
        
        if intent == "STORE_FACT":
            fact = self.memory.add(user_input)
            return {
                "response": f"✅ Learned: {fact}",
                "reasoning": "Action: INGESTION via MemoryLayer"
            }
        
        elif intent == "PROCUREMENT_REQUEST" or intent == "CHAT":
            try:
                # Use Runner to execute
                # user_input is a string, but Runner expects an object with .role and .parts
                # .parts must be a list of objects/dicts, e.g. [{"text": "..."}] OR objects with .text
                part_obj = SimpleNamespace(text=user_input)
                message_obj = SimpleNamespace(role="user", parts=[part_obj])

                response_generator = self.runner.run(
                    user_id="default_user",
                    session_id="default_session",
                    new_message=message_obj
                )
                
                # The runner returns a generator of events. We need to iterate to get the final answer.
                response_text = ""
                for event in response_generator:
                    logging.info(f"[DEBUG EVENT] {event}")
                    # Depending on event type (e.g. ModelResponse), accumulate text
                    try:
                        # Standard ADK event often has .content or .text
                        if hasattr(event, 'content'):
                             # content might be an object or string
                             c = event.content
                             if isinstance(c, str):
                                 response_text += c
                             elif hasattr(c, 'parts'):
                                 # If content is a message with parts
                                 for p in c.parts:
                                     if hasattr(p, 'text'):
                                         response_text += p.text or ""
                        elif hasattr(event, 'text'):
                             response_text += event.text or ""
                    except Exception as e:
                        logging.error(f"Error parse event: {e}")
                
                # If response_text is empty, try str(event) on the last event?
                # Actually, let's just inspect the event structure if this fails.
                # But 'response_obj' approach was wrong since it's a generator.
                
                # ... existing extraction logic ...
                log = f"ADK Agent Response:\n{response_text}"
                
                json_data = self.extract_json(response_text)
                if json_data:
                    status = json_data.get("status")
                    if status == "PAUSE_APPROVAL_NEEDED":
                        self.pending_order = json_data
                        return {
                            "response": f"⚠️ **Approval Required**\nReason: {json_data.get('reasoning')}\nCost: {json_data.get('total_cost')}\nProceed? (y/n)",
                            "reasoning": log
                        }
                    else:
                        return {
                            "response": f"✅ **Order Placed**\nVendor: {json_data.get('selected_vendor')}\nCost: {json_data.get('total_cost')}\nReason: {json_data.get('reasoning')}",
                            "reasoning": log
                        }
                else:
                    return {
                        "response": response_text,
                        "reasoning": log
                    }

            except Exception as e:
                logging.error(f"ADK Execution Error: {e}")
                return {"response": f"Error: {e}", "reasoning": str(e)}

    def handle_approval(self, user_input: str):
        if user_input.lower() in ["yes", "y", "approve", "ok"]:
            order = self.pending_order
            self.pending_order = None
            return {
                "response": f"✅ **Order Confirmed**.\nPlaced order with {order.get('selected_vendor')}.",
                "reasoning": "User manually approved."
            }
        else:
            self.pending_order = None
            return {
                "response": "❌Order Cancelled.",
                "reasoning": "User denied."
            }

    def extract_json(self, text):
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            return None
        return None
