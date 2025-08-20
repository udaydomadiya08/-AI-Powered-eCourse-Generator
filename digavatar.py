import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import base64
from io import BytesIO
from PIL import Image
import requests
from nltk.tokenize import sent_tokenize
from collections import defaultdict



import spacy
import whisperx

from Wav2Lip.inference import parser, run_inference

import os
import time
import random
import json
from datetime import datetime
from pathlib import Path

import os
import json
import random
import time
from datetime import datetime
import google.generativeai as genai
from collections import defaultdict

class GeminiResponse:
    def __init__(self, text):
        self.text = text

FAILED_KEYS_FILE = "disabled_keys.json"
USAGE_FILE = "usage_counts.json"
DAILY_LIMIT = 500
PER_MINUTE_LIMIT = 10

# In-memory per-minute usage tracker
minute_usage_tracker = defaultdict(list)

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

    # Daily count
    usage = load_json_file(USAGE_FILE)
    if today not in usage:
        usage[today] = {}
    if api_key not in usage[today]:
        usage[today][api_key] = 0
    usage[today][api_key] += 1
    save_json_file(USAGE_FILE, usage)

    # Per-minute in memory
    minute = now.strftime("%Y-%m-%d %H:%M")
    minute_usage_tracker[api_key] = [ts for ts in minute_usage_tracker[api_key] if ts.startswith(minute)]
    minute_usage_tracker[api_key].append(now.strftime("%Y-%m-%d %H:%M:%S"))

def has_exceeded_daily_limit(api_key, limit=DAILY_LIMIT):
    today = datetime.now().strftime("%Y-%m-%d")
    usage = load_json_file(USAGE_FILE)
    return usage.get(today, {}).get(api_key, 0) >= limit


def has_exceeded_minute_limit(api_key, limit=PER_MINUTE_LIMIT):
    current_minute = datetime.now().strftime("%Y-%m-%d %H:%M")
    recent_times = [ts for ts in minute_usage_tracker[api_key] if ts.startswith(current_minute)]
    return len(recent_times) >= limit


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

    model_names = [
        "gemini-2.5-flash-preview-05-20"
    ]

    disabled_keys_today = load_disabled_keys()

    for attempt in range(max_retries):
        available_keys = [
            k for k in api_keys
            if k not in disabled_keys_today and not has_exceeded_daily_limit(k) and not has_exceeded_minute_limit(k)
        ]

        if not available_keys:
            raise RuntimeError("❌ All API keys are either disabled or have reached daily/minute limits.")

        key = random.choice(available_keys)
        model = model_name or random.choice(model_names)

        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            gemini = genai.GenerativeModel(model)
            print(f"✅ Using model: {model}, key ending in: {key[-6:]}")
            
            response = gemini.generate_content(prompt)
            increment_usage(key)
            return GeminiResponse(response.text.strip())

        except Exception as e:
            print(f"❌ API call failed with key {key[-6:]} (Attempt {attempt + 1}): {e}")
            save_disabled_key(key)
            time.sleep(wait_seconds)

    raise RuntimeError("❌ All Gemini API attempts failed after retries.")
import json
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Path to persistent conversation log
DATA_FILE = "user_conversations.json"

# Load or create conversation log
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        conversation_log = json.load(f)
else:
    conversation_log = []

def save_conversation(user_msg, ai_msg):
    conversation_log.append({"user": user_msg, "ai": ai_msg})
    with open(DATA_FILE, "w") as f:
        json.dump(conversation_log, f, indent=2)

# Find best-matching past user messages
def get_relevant_context(user_input, top_k=5):
    if not conversation_log:
        return ""
    
    corpus = [chat['user'] for chat in conversation_log]
    vectorizer = TfidfVectorizer().fit_transform(corpus + [user_input])
    similarity_matrix = cosine_similarity(vectorizer[-1], vectorizer[:-1])
    
    top_indices = similarity_matrix.argsort()[0][-top_k:]
    context = ""
    for idx in top_indices:
        context += f"User: {conversation_log[idx]['user']}\nAI: {conversation_log[idx]['ai']}\n"
    return context

# Generate AI response using your Gemini wrapper
def respond_with_gemini(user_input):
    context = get_relevant_context(user_input)
    prompt = "Always reply in the user's style based on past conversations.\n" + context + f"User: {user_input}\nAI:"
    
    response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")
    ai_message = response.text.strip()
    save_conversation(user_input, ai_message)
    return ai_message

# Chat loop
while True:
    user_input = input("You: ")
    ai_response = respond_with_gemini(user_input)
    print("AI:", ai_response)
