from flask import Flask, request, jsonify
from datetime import datetime
from finrobot.agents.workflow import SingleAssistant
from finrobot.utils import get_current_date
import autogen

app = Flask(__name__)

# Load your OpenAI config (adjust the path if needed)
llm_config = {
    "config_list": autogen.config_list_from_json(
        "./OAI_CONFIG_LIST",  # Adjust path if needed
        filter_dict={"model": ["gpt-4-0125-preview"]},
    ),
    "timeout": 120,
    "temperature": 0,
}

# Instantiate the assistant (adjust name and config as needed)
assistant = SingleAssistant(
    "Market_Analyst",
    llm_config,
    human_input_mode="NEVER",
)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    messages = data.get("messages", [])
    user_message = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_message = m.get("content", "")
            break

    # Use the assistant to get a response
    try:
        # You can customize the prompt handling as needed
        response = assistant.chat(user_message)
    except Exception as e:
        response = f"Error: {str(e)}"

    return jsonify({
        "id": "chatcmpl-local-finrobot",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }
        ],
        "created": int(datetime.now().timestamp()),
        "model": "finrobot-local"
    })

if __name__ == "__main__":
    app.run(port=8000)
