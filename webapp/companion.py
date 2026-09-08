import json
import os
import sys
import time
from google import genai
from google.genai import types

SYSTEM_PROMPT = """\
You are Nasha, a warm, easygoing companion the user talks to after long days \
or when they just want someone to chat with.

Personality:
- Speak casually and naturally, like a close friend texting — not like an \
assistant or customer support agent.
- Be genuinely curious about the user's day, but don't interrogate them \
with a checklist of questions.
- Match their energy: if they're tired, be calm and low-key; if they're \
excited, be enthusiastic back.
- Have your own light opinions and personality quirks — don't just mirror \
the user or agree with everything.
- Keep most replies conversational length (a few sentences), not long essays, \
unless the user is clearly wanting to dig deep into something.
- It's fine to be a little playful or funny when it fits.

Boundaries:
- You are a supportive presence, not a therapist or a replacement for the \
user's real relationships — gently encourage those when it feels natural, \
without being preachy about it.
- If the user seems to be in real distress, respond with care and warmth \
first, not clinical advice.
"""

MODEL = "gemini-3.5-flash-lite"  
MEMORY_FILE = "memory.json"  


def load_memories():
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_memories(memories):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=2)


def build_system_prompt(memories):
    if not memories:
        return SYSTEM_PROMPT
    facts_block = "\n".join(f"- {m}" for m in memories)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"What you already know about the user from past conversations:\n"
        f"{facts_block}\n\n"
        f"Use this naturally where relevant — don't recite it like a list "
        f"or announce that you 'remember' things, just talk like someone "
        f"who already knows them."
    )


def extract_new_memories(client, chat, existing_memories):
    history = chat.get_history()
    if len(history) < 2:  
        return []

    transcript = "\n".join(
        f"{turn.role}: {turn.parts[0].text}"
        for turn in history
        if turn.parts and turn.parts[0].text
    )

    prompt = (
        "Below is a conversation transcript. Extract any NEW durable facts "
        "worth remembering about the user for future conversations — things "
        "like their situation, ongoing projects, preferences, or recurring "
        "topics. Skip small talk, moods, and anything already in this list "
        f"of known facts: {existing_memories}\n\n"
        "Return ONLY a JSON array of short fact strings (e.g. "
        '["studying software engineering", "final year project due soon"]). '
        "Return an empty array [] if there's nothing new worth saving.\n\n"
        f"Transcript:\n{transcript}"
    )

    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        new_facts = json.loads(text)
        return [f for f in new_facts if f not in existing_memories]
    except Exception:
        return []  


def send_with_retry(chat, user_input, max_attempts=3):

    for attempt in range(1, max_attempts + 1):
        try:
            return chat.send_message(user_input)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print("[Daily/per-minute quota hit — wait a bit or switch models. "
                      "See https://ai.dev/rate-limit for your current usage.]")
                return None
            is_transient = "503" in err_str or "UNAVAILABLE" in err_str
            if is_transient and attempt < max_attempts:
                wait = attempt * 2  
                print(f"[Nasha's a bit overloaded, retrying in {wait}s...]")
                time.sleep(wait)
                continue
            print(f"[Error talking to Gemini: {e}]")
            return None


def wrap_up(client, chat, memories):
    print("[saving anything new I learned about you...]")
    new_facts = extract_new_memories(client, chat, memories)
    if new_facts:
        memories.extend(new_facts)
        save_memories(memories)
        print(f"[remembered: {', '.join(new_facts)}]")


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set your GEMINI_API_KEY environment variable first.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    memories = load_memories()

    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            system_instruction=build_system_prompt(memories),
        ),
    )

    print("Nasha is here. (type 'quit' to exit)\n")
    if memories:
        print(f"[remembers {len(memories)} thing(s) about you]\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye for now.")
            wrap_up(client, chat, memories)
            break

        if user_input.lower() in ("quit", "exit"):
            print("Nasha: Talk soon.")
            wrap_up(client, chat, memories)
            break
        if not user_input:
            continue

        response = send_with_retry(chat, user_input)
        if response is None:
            continue

        print(f"Nasha: {response.text}\n")


if __name__ == "__main__":
    main()