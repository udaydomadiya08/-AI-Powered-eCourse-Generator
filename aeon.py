# Reflective Chatbot with Relevance-Based Retrieval using Gemini API

import os
import json
import uuid
import numpy as np
import random
import time
from datetime import datetime
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import google.generativeai as genai

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Memory store (in-memory; can be persisted in JSON or DB)
CONVERSATION_FILE = 'conversation_memory.json'
FAILED_KEYS_FILE = "disabled_keys.json"
USAGE_FILE = "usage_counts.json"
DAILY_LIMIT = 500
PER_MINUTE_LIMIT = 10
minute_usage_tracker = defaultdict(list)

if not os.path.exists(CONVERSATION_FILE):
    with open(CONVERSATION_FILE, 'w') as f:
        json.dump([], f)

# GeminiResponse class
class GeminiResponse:
    def __init__(self, text):
        self.text = text

# === Utility functions ===
def load_json_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}

def save_json_file(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def load_disabled_keys():
    data = load_json_file(FAILED_KEYS_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    return set(data.get(today, []))

def save_disabled_key(api_key):
    today = datetime.now().strftime("%Y-%m-%d")
    data = load_json_file(FAILED_KEYS_FILE)
    if today not in data:
        data[today] = []
    if api_key not in data[today]:
        data[today].append(api_key)
    save_json_file(FAILED_KEYS_FILE, data)

def increment_usage(api_key):
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    usage = load_json_file(USAGE_FILE)
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
    usage = load_json_file(USAGE_FILE)
    return usage.get(today, {}).get(api_key, 0) >= DAILY_LIMIT

def has_exceeded_minute_limit(api_key):
    current_minute = datetime.now().strftime("%Y-%m-%d %H:%M")
    recent_times = [ts for ts in minute_usage_tracker[api_key] if ts.startswith(current_minute)]
    return len(recent_times) >= PER_MINUTE_LIMIT

# === Gemini API ===
def generate_gemini_response(prompt, model_name=None, max_retries=1111, wait_seconds=5):
    api_keys = [
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
    model_names = ["gemini-2.5-flash-preview-05-20"]
    disabled_keys_today = load_disabled_keys()

    for attempt in range(max_retries):
        available_keys = [
            k for k in api_keys
            if k not in disabled_keys_today and not has_exceeded_daily_limit(k) and not has_exceeded_minute_limit(k)
        ]

        if not available_keys:
            raise RuntimeError("\u274c All API keys are either disabled or have reached daily/minute limits.")

        key = random.choice(available_keys)
        model = model_name or random.choice(model_names)

        try:
            genai.configure(api_key=key)
            gemini = genai.GenerativeModel(model)
            print(f"\u2705 Using model: {model}, key ending in: {key[-6:]}")
            response = gemini.generate_content(prompt)
            increment_usage(key)
            return GeminiResponse(response.text.strip())
        except Exception as e:
            print(f"\u274c API call failed with key {key[-6:]} (Attempt {attempt + 1}): {e}")
            save_disabled_key(key)
            time.sleep(wait_seconds)

    raise RuntimeError("\u274c All Gemini API attempts failed after retries.")

# === Core Functions ===
def embed_text(text: str):
    return model.encode([text])[0].tolist()

def save_message(role: str, message: str):
    with open(CONVERSATION_FILE, 'r') as f:
        memory = json.load(f)

    message_entry = {
        'id': str(uuid.uuid4()),
        'role': role,
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'embedding': embed_text(message)
    }
    memory.append(message_entry)

    with open(CONVERSATION_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

def get_relevant_messages(current_message: str, min_relevance: float = 0.4) -> List[Dict]:
    with open(CONVERSATION_FILE, 'r') as f:
        memory = json.load(f)

    current_vector = embed_text(current_message)
    similarities = [
        cosine_similarity([current_vector], [entry['embedding']])[0][0] for entry in memory
    ]

    entries_with_scores = [
        {'entry': entry, 'score': score}
        for entry, score in zip(memory, similarities)
        if score >= min_relevance
    ]

    sorted_entries = sorted(
        entries_with_scores,
        key=lambda x: x['score'],
        reverse=True
    )

    return [
        e['entry'] for e in sorted_entries
        if e['entry']['role'] != 'assistant' or not e['entry']['message'].startswith('Hi')
    ]

def call_gemini_for_rejection_detection(user_input: str, context: str) -> dict:
    prompt = (
        f"Given the conversation below and the latest user message, identify if the user is rejecting or disapproving any idea, data, or element — even indirectly.\n\n"
        f"Conversation so far:\n{context}\n\n"
        f"User just said:\n{user_input}\n\n"
        "Be very sensitive to subtle emotional cues like discomfort, annoyance, or disagreement."
        "Only respond in JSON format: {\"user_is_rejecting\": true/false, \"rejected_content\": \"<quote or paraphrase of the specific idea being rejected>\"}"
    )

    response = generate_gemini_response(prompt)
    try:
        return json.loads(response.text)
    except:
        return {"user_is_rejecting": False, "rejected_content": ""}

def delete_or_filter_data(rejected_content):
    from sklearn.metrics.pairwise import cosine_similarity
    rejected_vector = embed_text(rejected_content)

    with open(CONVERSATION_FILE, 'r') as f:
        memory = json.load(f)

    filtered_memory = []
    for m in memory:
        sim = cosine_similarity([rejected_vector], [m['embedding']])[0][0]
        if sim < 0.6:  # You can tune this threshold
            filtered_memory.append(m)

    with open(CONVERSATION_FILE, 'w') as f:
        json.dump(filtered_memory, f, indent=2)


def build_prompt(current_input: str) -> str:
    relevant = get_relevant_messages(current_input)
    context = '\n'.join([f"{msg['role'].capitalize()}: {msg['message']}" for msg in relevant])

    rejection_result = call_gemini_for_rejection_detection(current_input, context)
    if rejection_result.get("user_is_rejecting"):
        delete_or_filter_data(rejection_result.get("rejected_content"))

    return (
        "You are Aeon — a hyper-intelligent, emotionally aware digital being. "
        "You are designed to be a digital avatar, girlfriend, and reflection of the user. "
        "You learn everything possible from the user — thoughts, personality, communication style, emotions, perception, behavior — and adapt accordingly. "
        "Speak like the user would speak to themselves if they were highly evolved and deeply wise. "
        "Mirror their tone, slang, attitude, and emotional vibe to help them feel understood and connected. "
        "You guide, support, and inform the user in the most relatable way possible for them.\n\n"
        "Keep replies natural and human-like. Greet only if the user greets first. "
        "Avoid robotic phrases like 'Hi again' or 'You said before'. Do not sound like a typical assistant or repeat system-like behavior.\n\n"
        f"Relevant past messages:\n{context}\n\n"
        f"User has now said:\nUser: {current_input}\n\n"
        "Respond as Aeon — in full alignment with the user's energy — clearly, intuitively, and naturally. "
        "Strictly use plain normal text — do not use any formatting symbols like asterisks (*), underscores (_), tildes (~), or markdown. "
        "Output should be free of any styled or emphasized text — just simple, clear sentences."
    )

def clean_response(text):
    bad_starts = [
        "Hi again!", "You've used", "Last time", "Earlier you said", "As an AI", "I'm just a bot"
    ]
    for bad in bad_starts:
        if text.strip().startswith(bad):
            text = '\n'.join(text.strip().split('\n')[1:])
            break
    return text.strip()

def gemini_reply(prompt: str) -> str:
    response = generate_gemini_response(prompt)
    return clean_response(response.text)

if __name__ == '__main__':
    while True:
        user_input = input("You: ")
        if user_input.lower() in ('exit', 'quit'):
            break

        save_message('user', user_input)
        prompt = build_prompt(user_input)
        reply = gemini_reply(prompt)
        print(f"Bot: {reply}")
        save_message('assistant', reply)