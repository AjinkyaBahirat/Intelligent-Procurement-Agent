import gradio as gr
import pandas as pd
from src.agent import ADKProcurementAgent

# Initialize Agent
agent = ADKProcurementAgent()

def chat_response(message, history):
    """
    Handler for Chat Interface.
    Returns: (updated_history, reasoning_text)
    """
    result = agent.process_message(message)
    response_text = result["response"]
    reasoning_text = result["reasoning"]
    return response_text, reasoning_text

def get_memory_df():
    """
    Fetches all memories and converts to DataFrame for display.
    """
    memories = agent.memory.get_all()
    if not memories:
        return pd.DataFrame(columns=["Fact", "Metadata"])
    
    data = []
    for m in memories:
        # Simplify metadata for display
        meta = m.get("metadata", {})
        ts = meta.get("timestamp", "")
        data.append({"Fact": m["fact"], "Timestamp": ts})
    return pd.DataFrame(data)

with gr.Blocks(title="Intelligent Procurement Agent") as demo:
    gr.Markdown("# 🏗️ Intelligent Procurement Agent")
    
    with gr.Row():
        # Left Column: Chat
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=500, type="messages")
            msg = gr.Textbox(placeholder="Type your message (e.g., 'Order cement' or 'Limit is 50k')...", label="User Input")
            clear = gr.Button("Clear Chat")

        # Right Column: Reasoning & Memory
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("🧠 Live Reasoning"):
                    reasoning_box = gr.Textbox(label="Agent's Thought Process", lines=20, interactive=False)
                
                with gr.TabItem("📚 Knowledge Base"):
                    refresh_btn = gr.Button("Refresh Memory")
                    memory_table = gr.DataFrame(value=get_memory_df, interactive=False, wrap=True)

    # Interactions
    # When user submits message: 
    # 1. User message added to chat
    # 2. Agent processes -> Response added to chat, Reasoning updated in box
    
    def user_turn(user_message, history):
        return "", history + [{"role": "user", "content": user_message}]

    def bot_turn(history):
        # Last message is user's
        user_message = history[-1]["content"]
        response, reasoning = chat_response(user_message, history)
        history.append({"role": "assistant", "content": response})
        return history, reasoning

    msg.submit(user_turn, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_turn, [chatbot], [chatbot, reasoning_box]
    )
    
    refresh_btn.click(fn=get_memory_df, inputs=[], outputs=[memory_table])
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
