import os
import json
import random
import time
from datetime import datetime
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# --------------------------- CONFIG ---------------------------
DATA_FILE = "conversation_log.json"
STYLE_FILE = "user_style.json"
MEMORY_FILE = "conversation_memory.json"
FAILED_KEYS_FILE = "disabled_keys.json"
USAGE_FILE = "usage_counts.json"

DAILY_LIMIT = 500
PER_MINUTE_LIMIT = 10
MAX_RETRIES = 5
TOP_K_MATCHES = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL_NAME = "gemini-2.5-flash-preview-05-20"

API_KEYS = [
        "AIzaSyCG-BB-0iP8bTiTJiT9ZgC5eJkzDftV28I",
        "AIzaSyBGpWydub8jBjKW_JM808Q57x_KSVg1Fxw",
        "AIzaSyD-7i3eVHY_tQBlLedDGUYb12tPm88F2bg",
        "AIzaSyCT-678mR3ur4beLyWJJ-QdWA8W8cHvWtM",
        "AIzaSyBnKryfOjV-XsR0tdXWdYv4MXnbvvh_QWU",
        "AIzaSyBenRCth2XXKL6BXh_gRtDAznPfbTd9t4k",
        "AIzaSyCG6iIVxuoPAwRC8FL0DMHhywAFg58vxbM",
        "AIzaSyCWyJeh999WPRt5Mf8hgAfT78hkl_oyy3I",
        "AIzaSyDQoF2-V-jPVinMIHIs4Dts8KPpXeL-5_E",
        "AIzaSyA5VLL-EFpKs2Z0iVdLLK6ir_n9-b1wtrc",
        "AIzaSyCRNFcI51fF1KoS3YbBnaGtFMLIhSqnaSs",
        "AIzaSyCkhmr6hYUKCQMVuNaVMwhUmfLIrvOMn7g",
        "AIzaSyBFZS7DX_wDWvWjln22G3zN2XjORuMJV5o",
        "AIzaSyDOFT9J2OlqyR2KhhMP9qBaE3LqeLQLaIc",
        "AIzaSyDDBqYMNprBSxs006y_Mjm-2iFsedqvyE4",
        "AIzaSyAv7KdGJul7xb5tCnx2bLqZEStXTTtY-NA",
        "AIzaSyAFn_8ws-tj-ix7R_MIvTg-REUZ-93riZo",
        "AIzaSyB9ESuSqJMbEnAdvBKxaGJsfTrkdvaobYc",
        "AIzaSyCqXHtA2dl3tUumG21cMwbhxQdVP9LzypY",
        "AIzaSyAQCRR1KSbgiF3OkXUOInZOntFw1VU4n4k",
        "AIzaSyChdyTOEFX11YlnDaLKMh7IAXA_OzxpWSg",
        "AIzaSyBcBKM39mCY2x2-90tId2LRRQbOzwefLpE",
        "AIzaSyAKrMt8nww_uDt0stvfdsI8TX6T_SdSPjE",
        "AIzaSyCVrV7x9kfe2PPA3zroFl9usejOo8ROIFI",
        "AIzaSyCf5ekkmLP-1sweyXaUYbbg7OQ6Sbjl8rY",
        "AIzaSyAU4f5q3_szq3llfmMP3coyDRjrRKy-llk",
        "AIzaSyDq7c5xRhNs0Fn4CQR3Yt-3pDOlW6IrmTc",
        "AIzaSyCeBqAU_8suws4TQq0nf0qo2bwJhiIF5g4"


    ]

# ---------------------- GLOBALS ------------------------------
minute_usage_tracker = defaultdict(list)
conversation_log = []
user_style = {}
memory = {}
embed_model = SentenceTransformer(EMBEDDING_MODEL)

# ---------------------- JSON HELPERS -------------------------
def load_json_file(filepath, default=None):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except:
            return default if default is not None else {}
    return default if default is not None else {}

def save_json_file(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

# ---------------------- USAGE TRACKING -----------------------
def load_disabled_keys():
    data = load_json_file(FAILED_KEYS_FILE, {})
    today = datetime.now().strftime("%Y-%m-%d")
    return set(data.get(today, []))

def save_disabled_key(api_key):
    today = datetime.now().strftime("%Y-%m-%d")
    data = load_json_file(FAILED_KEYS_FILE, {})
    if today not in data:
        data[today] = []
    if api_key not in data[today]:
        data[today].append(api_key)
    save_json_file(FAILED_KEYS_FILE, data)

def increment_usage(api_key):
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    usage = load_json_file(USAGE_FILE, {})
    if today not in usage:
        usage[today] = {}
    if api_key not in usage[today]:
        usage[today][api_key] = 0
    usage[today][api_key] += 1
    save_json_file(USAGE_FILE, usage)

    minute = now.strftime("%Y-%m-%d %H:%M")
    minute_usage_tracker[api_key] = [ts for ts in minute_usage_tracker[api_key] if ts.startswith(minute)]
    minute_usage_tracker[api_key].append(now.strftime("%Y-%m-%d %H:%M:%S"))

def has_exceeded_daily_limit(api_key):
    today = datetime.now().strftime("%Y-%m-%d")
    usage = load_json_file(USAGE_FILE, {})
    return usage.get(today, {}).get(api_key, 0) >= DAILY_LIMIT

def has_exceeded_minute_limit(api_key):
    current_minute = datetime.now().strftime("%Y-%m-%d %H:%M")
    recent_times = [ts for ts in minute_usage_tracker[api_key] if ts.startswith(current_minute)]
    return len(recent_times) >= PER_MINUTE_LIMIT

# ---------------------- GEMINI CALL --------------------------
def generate_gemini_response(prompt, model_name=None, max_retries=MAX_RETRIES, wait_seconds=5):
    disabled_keys_today = load_disabled_keys()
    model_name = model_name or MODEL_NAME

    for attempt in range(max_retries):
        available_keys = [
            k for k in API_KEYS
            if k not in disabled_keys_today and not has_exceeded_daily_limit(k) and not has_exceeded_minute_limit(k)
        ]
        if not available_keys:
            raise RuntimeError("❌ All API keys disabled or limit reached.")

        key = random.choice(available_keys)
        try:
            genai.configure(api_key=key)
            gemini = genai.GenerativeModel(model_name)
            print(f"✅ Using model: {model_name}, key ending in: {key[-6:]}")
            response = gemini.generate_content(prompt)
            increment_usage(key)
            return response.text.strip()
        except Exception as e:
            print(f"❌ Gemini call failed (Attempt {attempt + 1}): {e}")
            save_disabled_key(key)
            time.sleep(wait_seconds)
    raise RuntimeError("❌ All Gemini API attempts failed.")

# ---------------------- LOAD / INIT --------------------------
conversation_log = load_json_file(DATA_FILE, [])
user_style = load_json_file(STYLE_FILE, {})
# ---------------------- LOAD MEMORY -----------------------
def load_memory():
    memory = load_json_file(MEMORY_FILE, {})
    if not isinstance(memory, dict):
        memory = {}  # reset if corrupted
    memory.setdefault("history", [])  # everything user says
    memory.setdefault("tone", "neutral")  # optional tone tracking
    return memory

memory = load_memory()

# Ensure memory is valid before any update
def ensure_memory_keys(mem):
    if not isinstance(mem, dict):
        mem = load_memory()
    mem.setdefault("history", [])
    mem.setdefault("tone", "neutral")
    return mem

# ---------------------- MEMORY HANDLER -----------------------
def analyze_user_message(user_input):
    """
    Optional: detect tone changes or style dynamically.
    Returns JSON like {"update_tone": "excited"} or {}.
    """
    prompt = f"""
Detect if the user is expressing a tone change. Return JSON like:
{{"update_tone": "excited"}}
Message: {user_input}
"""
    try:
        response = generate_gemini_response(prompt)
        return json.loads(response)
    except:
        return {}

def update_memory(user_input):
    global memory
    memory = ensure_memory_keys(memory)

    # Append input to general-purpose history
    memory["history"].append(user_input)

    # Optional: detect tone changes
    actions = analyze_user_message(user_input)
    if "update_tone" in actions:
        memory["tone"] = actions["update_tone"]

    save_json_file(MEMORY_FILE, memory)


# ---------------------- CONTEXT SEARCH -----------------------
def find_best_matches(user_input, top_k=TOP_K_MATCHES):
    if not conversation_log:
        return []
    query_emb = embed_model.encode(user_input)
    scores = [(util.cos_sim(query_emb, entry["embedding"]).item(), entry) for entry in conversation_log]
    scores.sort(reverse=True, key=lambda x: x[0])
    return [entry for _, entry in scores[:top_k]]

# ---------------------- SAVE CONVERSATION -------------------
def save_conversation(user_msg, ai_msg):
    combined = user_msg + " " + ai_msg
    embedding = embed_model.encode(combined).tolist()
    conversation_log.append({
        "user": user_msg,
        "ai": ai_msg,
        "embedding": embedding
    })
    save_json_file(DATA_FILE, conversation_log)

# ---------------------- STYLE DETECTION ----------------------
def detect_user_style(user_message, style_data):
    if "call me " in user_message.lower():
        nickname = user_message.lower().split("call me ")[-1].strip()
        style_data.setdefault("nicknames", [])
        if nickname not in style_data["nicknames"]:
            style_data["nicknames"].append(nickname)

    style_data.setdefault("phrases", [])
    for w in user_message.split():
        if w not in style_data["phrases"]:
            style_data["phrases"].append(w)

    if "!" in user_message:
        style_data["tone"] = "excited"
    elif "?" in user_message:
        style_data["tone"] = "questioning"
    else:
        style_data.setdefault("tone", "neutral")

    save_json_file(STYLE_FILE, style_data)
    return style_data

# ---------------------- BUILD PROMPT ------------------------
def build_prompt(user_input):
    """
    Build the AI prompt using memory so it can recall previous user info dynamically.
    """
    global memory
    memory = ensure_memory_keys(memory)

    # Extract potential name from history (simple heuristic)
    name = None
    for msg in reversed(memory["history"]):
        if "my name is " in msg.lower():
            name = msg.lower().split("my name is ")[-1].strip()
            break

    # Add memory info to the prompt
    memory_text = f"User's name: {name}\n" if name else ""
    memory_text += f"Memory history: {', '.join(memory['history'][-10:])}\n"  # last 10 messages

    instructions = (
        "Respond naturally to the user's latest message.\n"
        "Use memory context subtly to answer questions about the user.\n"
    )

    return f"{instructions}{memory_text}User: {user_input}\nAI:"

# ---------------------- CHAT LOOP ---------------------------
def chat():
    print("=== Advanced Context-Aware AI Avatar (Gemini) ===")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Update memory with user input
        update_memory(user_input)

        # Build prompt with memory context
        prompt = build_prompt(user_input)

        # Generate AI response
        ai_response = generate_gemini_response(prompt)
        print(f"AI: {ai_response}")

        # Save conversation
        save_conversation(user_input, ai_response)


# ---------------------- RUN ---------------------------------
if __name__ == "__main__":
    chat()
