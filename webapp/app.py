import os

from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types

from companion import (
    MODEL,
    build_system_prompt,
    extract_new_memories,
    load_memories,
    save_memories,
    send_with_retry,
)

app = Flask(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Set GEMINI_API_KEY as an environment variable before starting.")

client = genai.Client(api_key=api_key)
memories = load_memories()

chat = client.chats.create(
    model=MODEL,
    config=types.GenerateContentConfig(system_instruction=build_system_prompt(memories)),
)

message_count = 0
SAVE_MEMORY_EVERY = 5  


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    global message_count

    data = request.get_json(force=True) or {}
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "empty message"}), 400

    response = send_with_retry(chat, user_message)
    if response is None:
        return jsonify({"error": "Couldn't get a response right now — try again."}), 502

    message_count += 1
    if message_count % SAVE_MEMORY_EVERY == 0:
        new_facts = extract_new_memories(client, chat, memories)
        if new_facts:
            memories.extend(new_facts)
            save_memories(memories)

    return jsonify({"reply": response.text})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)