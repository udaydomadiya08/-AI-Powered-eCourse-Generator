import os
import base64
from io import BytesIO
from PIL import Image
import requests
from nltk.tokenize import sent_tokenize

from moviepy.editor import (
    AudioFileClip, ImageClip, CompositeVideoClip,
    concatenate_videoclips, concatenate_audioclips,
    TextClip
)

from moviepy.video.fx.all import resize
from moviepy.video.tools.subtitles import SubtitlesClip

import spacy
import whisperx

from Wav2Lip.inference import parser, run_inference

import os
import time
import random
import json
from datetime import datetime
from pathlib import Path

class GeminiResponse:
    def __init__(self, text):
        self.text = text

FAILED_KEYS_FILE = "disabled_keys.json"

def load_disabled_keys():
    if not os.path.exists(FAILED_KEYS_FILE):
        return set()

    with open(FAILED_KEYS_FILE, "r") as f:
        data = json.load(f)
    today = datetime.now().strftime("%Y-%m-%d")
    return set(data.get(today, []))

def save_disabled_key(api_key):
    today = datetime.now().strftime("%Y-%m-%d")
    data = {}

    if os.path.exists(FAILED_KEYS_FILE):
        with open(FAILED_KEYS_FILE, "r") as f:
            data = json.load(f)

    if today not in data:
        data[today] = []

    if api_key not in data[today]:
        data[today].append(api_key)

    with open(FAILED_KEYS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def generate_gemini_response(prompt, model_name=None, max_retries=7, wait_seconds=5):
    import google.generativeai as genai

    api_keys = [
        "AIzaSyA2Hj5phmEsqXBWqIGbZxQXxAzv129Zw1E",
        "AIzaSyDwTEr-7c2kP7doddq93aG9CpRmiz0Bv44",
        "AIzaSyAvsg_Oky2NJpD3uNnqMHF4xQJRBK3V9RY",
        "AIzaSyArpDip4G3DK3MiiN_mwE6CHpgDRtQD9TU",
        "AIzaSyBRzQCetzqXL9aQDcQw8T2C0rnzRxIYTTw",
        "AIzaSyCL46bkyk5tvCZNJCsAA3VSf-NC-g7BU3o",
        "AIzaSyD4Pv1UiLc7fsA7InuOvhhWxVkMgGAO8dI",
        "AIzaSyCeBdgElggdHYaHnf4N3z0RlPeROZ5LzEU"
    ]

    model_names = [
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-preview-05-06"
    ]

    disabled_keys_today = load_disabled_keys()

    for attempt in range(max_retries):
        available_keys = [k for k in api_keys if k not in disabled_keys_today]
        if not available_keys:
            raise RuntimeError("❌ All Gemini API keys are disabled for today.")

        key = random.choice(available_keys)
        model = model_name or random.choice(model_names)

        try:
            genai.configure(api_key=key)
            gemini = genai.GenerativeModel(model)
            print(f"✅ Success with model: {model}, key: {key[-6:]}")

            response = gemini.generate_content(prompt)
            return GeminiResponse(response.text.strip())

        except Exception as e:
            print(f"❌ Failed with key {key[-6:]} (Attempt {attempt + 1}): {e}")
            save_disabled_key(key)
            disabled_keys_today.add(key)
            time.sleep(wait_seconds)

    raise RuntimeError("❌ All Gemini API attempts failed.")


import sys
import os
sys.path.append(os.path.abspath('Wav2Lip'))

def run_wav2lip_inference(
    checkpoint_path,
    face_video,
    audio_path,
    output_video,
    static=True,
    fps=24,
    wav2lip_batch_size=128,   # default value
    resize_factor=1,
    out_height=480
):
    args_list = [
        '--checkpoint_path', checkpoint_path,
        '--face', face_video,
        '--audio', audio_path,
        '--outfile', output_video,
        '--fps', str(fps),
        '--wav2lip_batch_size', str(wav2lip_batch_size),
        '--resize_factor', str(resize_factor),
        '--out_height', str(out_height),
    ]

    if static:
        args_list.append('--static')

    args = parser.parse_args(args_list)

    print("Starting Wav2Lip inference...")
    run_inference(args)
    print("Inference done!")

# === Setup ===
nlp = spacy.load("en_core_web_sm")  # Load spaCy model once

# Gemini API keys (replace with your keys)
api_keys = [
    "AIzaSyCeBdgElggdHYaHnf4N3z0RlPeROZ5LzEU",
    "AIzaSyD4Pv1UiLc7fsA7InuOvhhWxVkMgGAO8dI",
    "AIzaSyDiFNqDfLPHbK4JzciGHKvD9JijTqxIbtE",
    "AIzaSyBFZS7DX_wDWvWjln22G3zN2XjORuMJV5o",
    "AIzaSyCL46bkyk5tvCZNJCsAA3VSf-NC-g7BU3o",
    "AIzaSyDOFT9J2OlqyR2KhhMP9qBaE3LqeLQLaIc",
    "AIzaSyDwTEr-7c2kP7doddq93aG9CpRmiz0Bv44",
    "AIzaSyA2Hj5phmEsqXBWqIGbZxQXxAzv129Zw1E",
    "AIzaSyCkhmr6hYUKCQMVuNaVMwhUmfLIrvOMn7g",
    "AIzaSyCRNFcI51fF1KoS3YbBnaGtFMLIhSqnaSs",
    "AIzaSyA5VLL-EFpKs2Z0iVdLLK6ir_n9-b1wtrc",
    "AIzaSyAX8dLULAq6MlCeo8rg0oP0YoTV6FAfqxE",
    "AIzaSyD_Uraq02m1Xk0K9CbWZNFYleoilHcoaB8",
    "AIzaSyDQoF2-V-jPVinMIHIs4Dts8KPpXeL-5_E",
    "AIzaSyCG6iIVxuoPAwRC8FL0DMHhywAFg58vxbM",
    "AIzaSyCWyJeh999WPRt5Mf8hgAfT78hkl_oyy3I"


]

# Google TTS service account json path
TTS_JSON_KEY_PATH = "my-project-tts-461911-dbd39de52028.json"

# Video size & font for subtitles
VIDEO_SIZE = (1080, 1920)
FONT_PATH = "/Users/uday/Downloads/VIDEOYT/Anton-Regular.ttf"
FONT_SIZE = 110

# Create needed directories once
os.makedirs("audio", exist_ok=True)
os.makedirs("temp/images", exist_ok=True)
os.makedirs("video_created", exist_ok=True)

# === Gemini Image Generation ===
from google import genai
from google.genai import types

def resize_to_1080x1920_stretch(image: Image.Image) -> Image.Image:
    return image.resize(VIDEO_SIZE, Image.LANCZOS)

def compress_image(input_image: Image.Image, output_path: str, max_size_kb=2048):
    quality = 90
    while quality >= 20:
        input_image.save(output_path, format="JPEG", quality=quality)
        if os.path.getsize(output_path) <= max_size_kb * 1024:
            break
        quality -= 5
    return output_path

def generate_image_for_sentence(topic: str, sentence: str, image_index: int = 0) -> str:
    prompt = (
        "Generate a high-resolution vertical image (1080x1920) that visually represents the following concept "
        "within the context of the topic:\n"
        f"📌 Topic: '{topic}'\n"
        f"💡 Idea: \"{sentence}\"\n\n"
        "The image should be cinematic, clean, and highly professional. Use vibrant colors, sharp details, and clear lighting. "
        "Avoid any text, watermarks, or logos. The visual should strongly reflect both the topic and the idea, making it suitable "
        "for inclusion in a modern YouTube video. Ensure it looks polished, emotionally engaging, and scroll-stopping in quality."
        "mind it do not overlay snetenc or topic text on image, i dont want topic or sentence text on my image mind it i only want visual image"
    )

    response = None
    for key in api_keys:
        try:
            print(f"🔁 Trying Gemini API key ending with: {key[-6:]}")
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=[prompt],  # contents as list (API usually requires list)
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            break
        except Exception as e:
            print(f"❌ Gemini API key failed: {e}")
            continue

    if not response:
        print("❌ All Gemini API keys failed.")
        return None

    # Parse image data from response
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            data = part.inline_data.data
            image_data = None

            if isinstance(data, str):
                try:
                    image_data = base64.b64decode(data)
                except Exception as e:
                    print("⚠️ Base64 decode error:", e)
            elif isinstance(data, (bytes, bytearray)):
                image_data = data

            if image_data:
                try:
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                    image = resize_to_1080x1920_stretch(image)
                    output_path = f"temp/images/img_{image_index}.jpg"
                    compress_image(image, output_path)
                    print(f"✅ Image saved: {output_path}")
                    return output_path
                except Exception as e:
                    print("❌ Error processing image:", e)
                    return None

    print("❌ No image data found in Gemini response.")
    return None

def generate_youtube_shorts_script(topic):
    prompt = f"""
    Write a YouTube Shorts script about "{topic}" in a high-energy, fast-paced, no-fluff style. create best stongest hook to keep viewers engaged in video, then scrip tmust too be very engaging and high retention one so user dont move shortt to nex tmind it that highest lvel of engagment must be done viewwer shoul dsee full video mind it writet that level of script. The script must be delivered as one single paragraph of plain text — no formatting, no headings, no bullet points, no labels — just pure spoken content. The tone should be exciting, direct, and packed with value, instantly hooking the viewer and keeping them engaged throughout. 
    video script msut be engaging  so best high level that user tend to see full video and dont leave in betweeen, it should be educatiave informative fro user infintly valuable fro user so user can be highly benefitted. user is beyond god for us mind it. -we are making videos for audience and not robots so give script likwise we need to serve our audience mind it
    Keep the message clear and compelling with short, punchy sentences designed for maximum impact. Avoid intros or outros — just jump straight into the content and keep the energy high. Make it sound like a rapid-fire narration by a confident creator. i want script that gtts can understand because gtts canno tunderstand short forms avoid any ytpe of short forms use only where necessary, like(example, use is not instaed of isn't) so dont use unnecessary short froms only use when it is neccessary fro specific word shortforms like (RIP etc)
    entire script must be hooky, hook throughout video script write like that, mind it very important, user msut see full video  viewwer must not left my video unseen viewer should must see my full video short, viewer must becmoe a apart of my channel by likig sharing subscribing commenting good hings write ethat level of script i wan to earn infinte money pls help me with oyur video script i beg you pls write that level of scritp pls mind ti i want like that please request.
    The final script must not exceed what can be spoken in 55–60 seconds using gTTS (around 200-250 words, or ~16-18 concise sentences). This will be used by an AI video generator, so keep formatting clean and the paragraph tight. Prioritize viral potential, monetization value, and viewer retention. at end tell user to like comment and subscribe to my channel tell it dont ask it if they liked then do just tell them to do all 3 at end, okay mind it,  mind it it should bes seo optimised
    dont inlude astreiks bold text only plain format normal text mind it
    """
    response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")
    return response.text.strip()
import ffmpeg
def mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 to WAV using ffmpeg"""
    if os.path.exists(wav_path):
        os.remove(wav_path)
        print(f"🗑️ Deleted existing WAV file: {wav_path}")

    try:
        ffmpeg.input(mp3_path).output(wav_path, ar=16000, ac=1).run(overwrite_output=True)
        print(f"✅ MP3 to WAV Conversion complete: {wav_path}")
    except ffmpeg.Error as e:
        print("❌ FFmpeg error during mp3_to_wav:")
        print(e.stderr.decode())



def wav_to_mp3(wav_path, mp3_path):
    """Convert WAV to MP3 using ffmpeg"""
    if os.path.exists(mp3_path):
        os.remove(mp3_path)
        print(f"🗑️ Deleted existing MP3 file: {mp3_path}")

    try:
        ffmpeg.input(wav_path).output(mp3_path, acodec='libmp3lame').run(overwrite_output=True)
        print(f"✅ WAV to MP3 Conversion complete: {mp3_path}")
    except ffmpeg.Error as e:
        print("❌ FFmpeg error during wav_to_mp3:")
        print(e.stderr.decode())


# === Google Text-to-Speech ===
from google.oauth2 import service_account
from google.auth.transport.requests import Request

def generate_tts_audio(text, filename="output.mp3", json_key_path=TTS_JSON_KEY_PATH,
                       voice_name="en-US-Studio-O", speaking_rate=1.4):
    # Set credentials env once (optional, but safe)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path

    credentials = service_account.Credentials.from_service_account_file(
        json_key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    access_token = credentials.token

    url = "https://texttospeech.googleapis.com/v1/text:synthesize"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {"text": text},
        "voice": {"languageCode": "en-US", "name": voice_name},
        "audioConfig": {"audioEncoding": "MP3", "speakingRate": speaking_rate}
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        audio_content = response.json()["audioContent"]
        with open(filename, "wb") as out:
            out.write(base64.b64decode(audio_content))
        print(f"✅ Audio saved to {filename}")
        return filename
    else:
        print(f"❌ TTS error {response.status_code}: {response.text}")
        return None

# === WhisperX subtitle creation using SubtitlesClip ===
from moviepy.editor import TextClip, CompositeVideoClip
from moviepy.video.fx.all import fadein, fadeout, resize

import numpy as np
import random
from moviepy.editor import VideoClip, CompositeVideoClip, TextClip
from moviepy.video.fx.all import fadein, fadeout, resize
import colorsys

def create_gradient_frame(w, h, offset, direction, left_color, right_color):
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)

    if direction == "diagonal_tl_br":
        pos = (X + Y) / 2
        ratio = (pos + offset) % 1

    elif direction == "diagonal_br_tl":
        pos = (X + Y) / 2
        ratio = (1 - pos + offset) % 1

    else:
        ratio = np.full((h, w), offset)

    gradient = (1 - ratio[..., None]) * left_color + ratio[..., None] * right_color
    return gradient.astype(np.uint8)

def hsv_to_rgb_array(h, s, v):
    """Convert arrays of HSV to RGB arrays (values 0-1)"""
    import colorsys
    rgb = np.array([colorsys.hsv_to_rgb(h_, s, v) for h_ in h.flatten()])  # (N,3)
    return rgb.reshape(h.shape + (3,))

def random_bright_color():
    # Return bright HSV (random hue, full saturation & value)
    h = random.random()
    s = 1.0
    v = 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return np.array([r, g, b]) * 255

def create_word_gradient_clip(word, duration, font, fontsize, video_size):
    text_clip = TextClip(word, fontsize=fontsize, font=font, color="white", method="label")

    text_mask = text_clip.to_mask()
    w, h = text_mask.size

    left_color = random_bright_color()
    right_color = random_bright_color()
    direction = random.choice(["diagonal_tl_br", "diagonal_br_tl"])

    def make_frame(t):
        progress = (t / duration) % 1
        offset = progress
        gradient = create_gradient_frame(w, h, offset, direction, left_color, right_color)
        mask_frame = text_mask.get_frame(t)
        colored_frame = (gradient * mask_frame[:, :, None]).astype(np.uint8)
        return colored_frame

    return VideoClip(make_frame, duration=duration).set_position("center").set_mask(text_mask)


def create_word_by_word_subtitles(
    word_segments,
    video_size=(1080, 1920),
    font="/Users/uday/Downloads/VIDEOYT/Anton-Regular.ttf",
    fontsize=110,
):
    from moviepy.editor import CompositeVideoClip
    from moviepy.video.fx.all import fadein, fadeout

    clips = []

    for word_info in word_segments:
        word = word_info["word"]
        start = word_info["start"]
        end = word_info["end"]
        duration = end - start

        grad_clip = create_word_gradient_clip(word, duration, font, fontsize, video_size)
        grad_clip = grad_clip.set_start(start).fx(fadein, 0.05).fx(fadeout, 0.05)
        clips.append(grad_clip)

    return CompositeVideoClip(clips, size=video_size)

# Load whisperx model once, outside to avoid reloading multiple times
device = "cpu"
print(f"Using device for WhisperX: {device}")

whisper_model = whisperx.load_model("tiny.en", device=device, compute_type="float32")
align_model, align_metadata = whisperx.load_align_model(language_code="en", device=device)


def create_scene_clip(sentence, image_path, audio_path, video_size=VIDEO_SIZE):
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration

    img_clip = ImageClip(image_path).set_duration(duration).resize(video_size).set_fps(30)

    # Transcribe & align audio for subtitles
    result = whisper_model.transcribe(audio_path)
    aligned_result = whisperx.align(result["segments"], align_model, align_metadata, audio_path, device)
    word_segments = aligned_result["word_segments"]

    subtitle_clip = create_word_by_word_subtitles(word_segments, video_size=video_size)

    scene = CompositeVideoClip([img_clip, subtitle_clip.set_duration(duration)])

    scene = scene.set_duration(duration).set_audio(None)
  # very important to avoid overlaps

    return scene

# === Main function to create video from script ===
from moviepy.editor import AudioFileClip, CompositeAudioClip, concatenate_videoclips, concatenate_audioclips

def create_video_from_script_with_whisperx(script, user_topic,output_path):
    import tempfile
    import subprocess
    import random
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, vfx
    from nltk.tokenize import sent_tokenize
    import os

    def save_clip_to_tempfile(clip, suffix="", pad_duration=0.08):
        clip = clip.set_fps(30)
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=f"{suffix}.mp4").name
        padded_path = tempfile.NamedTemporaryFile(delete=False, suffix=f"{suffix}_padded.mp4").name

        clip.write_videofile(temp_path, codec="libx264", audio_codec="aac", fps=30, preset="ultrafast", threads=8, logger=None)

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", temp_path,
            "-vf", f"tpad=stop_mode=clone:stop_duration={pad_duration}",
           
            "-preset", "ultrafast",
            padded_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        os.remove(temp_path)
        return padded_path

    def ffmpeg_chain_random_transitions(clip_paths, output_path, transition_duration=0.08):
        import tempfile
        import os
        import random
        from moviepy.editor import VideoFileClip
        import subprocess

        transitions = [
            "fade", "fadeblack", "fadewhite", "radial", "circleopen",
            "circleclose", "rectcrop", "distance", "slideleft", "slideright", "slideup", "slidedown"
        ]

        # If only one clip, just copy it to the output path
        if len(clip_paths) == 1:
            os.rename(clip_paths[0], output_path)
            return

        # First clip remains unchanged
        current_input = clip_paths[0]

        for i in range(1, len(clip_paths)):
            next_input = clip_paths[i]

            # Get durations
            duration_current = VideoFileClip(current_input).duration
            offset = duration_current - transition_duration

            transition_type = random.choice(transitions)
            print(f"🎞️ Transition {i}: '{transition_type}' at offset={offset:.2f}s")

            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", current_input,
                "-i", next_input,
                "-filter_complex",
                f"[0:v][1:v]xfade=transition={transition_type}:duration={transition_duration}:offset={offset},format=yuv420p[v]",
                "-map", "[v]",
                "-map", "0:a?",  # use audio from first
                "-c:a", "aac",
                "-preset", "ultrafast",
                "-crf", "23",
                temp_output
            ]

            subprocess.run(ffmpeg_cmd, check=True)

            if current_input not in clip_paths:
                try:
                    os.remove(current_input)
                except:
                    pass

            current_input = temp_output

        os.rename(current_input, output_path)


    # --- MAIN LOGIC START ---
    sentences = sent_tokenize(script)
    scene_clips = []

    print(f"🎥 Total scenes to generate: {len(sentences)}")

    for idx, sentence in enumerate(sentences):
        print(f"\n🔹 Generating Scene {idx + 1}: {sentence}")

        image_path = generate_image_for_sentence(user_topic, sentence, idx)
        if not image_path:
            print(f"❌ Failed to generate image for scene {idx + 1}")
            continue

        audio_path = f"audio/scene_{idx}.mp3"
        audio_file = generate_tts_audio(sentence, audio_path)
        if not audio_file or not os.path.exists(audio_file):
            print(f"❌ Failed to generate audio for scene {idx + 1}")
            continue

        scene_clip = create_scene_clip(sentence, image_path, audio_path)
        scene_clips.append(scene_clip)

    if not scene_clips:
        print("❌ No scenes generated successfully.")
        return None

    # Save each clip to padded temporary file
    transition_duration = 0.08
    temp_files = [
        save_clip_to_tempfile(
            clip.resize((1080, 1920)),
            suffix=f"_scene{i}",
            pad_duration=0 if i == 0 else transition_duration  # ⬅️ No padding for first scene
        )
        for i, clip in enumerate(scene_clips)
    ]


    # Apply transitions
    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    ffmpeg_chain_random_transitions(temp_files, output_temp, transition_duration)

    # Clean up temp files
    for f in temp_files:
        try:
            os.remove(f)
        except:
            pass

    final_video = VideoFileClip(output_temp)

    # Add background music - UPDATED TO AVOID AUDIO ISSUES
    try:
        from moviepy.audio.fx.all import audio_loop

        # Load background music and reduce volume
        bg_music_raw = AudioFileClip("/Users/uday/Downloads/VIDEOYT/Cybernetic Dreams.mp3").volumex(0.08)

        # Loop background music to match video duration and set start to 0 explicitly
        bg_music_looped = audio_loop(bg_music_raw, duration=final_video.duration).set_start(0)

        if final_video.audio:
            # Mix existing audio with background music
            final_audio_with_bg = CompositeAudioClip([
                final_video.audio.set_duration(final_video.duration).set_start(0),
                bg_music_looped
            ])
        else:
            # If no original audio, just use background music
            final_audio_with_bg = bg_music_looped

        # Set the combined audio back to video
        final_video = final_video.set_audio(final_audio_with_bg)

    except Exception as e:
        print(f"⚠️ Background music could not be applied: {e}")
        # Fallback: keep original audio if exists, else None
        final_video = final_video.set_audio(final_video.audio if final_video.audio else None)

  
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", audio=True, fps=30)

    print(f"\n✅ Final video created at: {output_path}")
    return output_path

from googleapiclient.http import MediaFileUpload


import os
import pickle
from googleapiclient.discovery import build

TOKEN_PATHS = [
    "/Users/uday/Downloads/VIDEOYT/token1.pickle",
    "/Users/uday/Downloads/VIDEOYT/token2.pickle",
    "/Users/uday/Downloads/VIDEOYT/token3.pickle",
    "/Users/uday/Downloads/VIDEOYT/token.pickle"
]

def build_youtube_client(token_path):
    with open(token_path, "rb") as token_file:
        credentials = pickle.load(token_file)
    return build("youtube", "v3", credentials=credentials)

def is_token_usable(youtube):
    try:
        channel_response = youtube.channels().list(part="contentDetails", mine=True).execute()
        return channel_response is not None
    except Exception as e:
        print(f"❌ Token not usable or quota exceeded: {e}")
        return False

def get_available_token():
    for path in TOKEN_PATHS:
        try:
            youtube = build_youtube_client(path)
            if is_token_usable(youtube):
                print(f"✅ Using token: {path}")
                return youtube, path
        except Exception as e:
            print(f"⚠️ Failed to load token: {path} — {e}")
    return None, None



import string

def sanitize_filename(name):
    valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
    return ''.join(c if c in valid_chars else '_' for c in name)

def generate_course_outline(topic):
    prompt = f"""
Generate a structured video course for the topic: "{topic}".

Structure it like this:
Chapter 1: [Chapter Title]
- Lesson 1: [Lesson Title]
- Lesson 2: [Lesson Title]

Chapter 2: [Chapter Title]
- Lesson 1: [Lesson Title]
...

only 1 chapter and only two lesson mind it it is fro testing purpose so reuire that only
mind it plain text dont use anyhting else bold text, etc. i want plain format text nothing else in the way given above output in same manner only plain text
"""
#Minimum: 3 chapters, each with 3-5 lessons.     
    try:
        response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")
        return response.text.strip()
    except Exception as e:
        print(f"Error generating course outline: {e}")
        return ""

import re

def parse_outline(raw_text):
    chapters = {}
    current_chapter = None

    # Match chapter lines like '**Chapter 1: Foundations of Machine Learning**'
    # Allow optional leading/trailing **
    chapter_pattern = re.compile(r'^\**\s*Chapter\s+(\d+)\s*:\s*(.+?)\s*\**$', re.IGNORECASE)

    # Match lesson lines like '- Lesson 1: Title'
    lesson_pattern = re.compile(r'^-\s*Lesson\s+\d+\s*:\s*(.+)$', re.IGNORECASE)

    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith('---') or line.startswith("Here's"):
            continue

        print(f"Parsing line: '{line}'")  # Keep this debug if you want

        chapter_match = chapter_pattern.match(line)
        if chapter_match:
            chap_num = int(chapter_match.group(1))
            title = chapter_match.group(2).strip()
            current_chapter = f"Chapter {chap_num}"
            chapters[current_chapter] = {"title": title, "lessons": []}
            continue

        lesson_match = lesson_pattern.match(line)
        if lesson_match and current_chapter:
            lesson_title = lesson_match.group(1).strip()
            chapters[current_chapter]["lessons"].append(lesson_title)

    return chapters



def generate_script(topic, chapter_title, lesson_title):
    prompt = f"""
Create a video lesson script (less than 300 words) for the following:

Course Topic: {topic}
Chapter Title: {chapter_title}
Lesson Title: {lesson_title}

Keep it educational, beginner-friendly, engaging, and use simple language. mind it plain text dont use anyhting else bold text, etc. i want plain format text nothing else in the way given above output in same manner only plain text. ouptu mus tin paragraph formt only i want only script and nothing else other than it like suggestion for visual scene voice anyhting else i dont want it only i want is script in paragraph fromat and nothing else and that to iin planitext only dont use bold text or anyhting like that just plain fromat text dont break to next line i want in continous manner. i want strictly in one line only dont move to next line only mind it
"""
    try:
        response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")
        return response.text.strip()
    except Exception as e:
        print(f"Error generating script for {lesson_title}: {e}")
        return "Script generation failed."

def save_lesson(course_dir, chapter_num, chapter_title, lesson_num, lesson_title, content):
    safe_lesson_title = sanitize_filename(lesson_title)
    filename = f"{lesson_num:02d}_{safe_lesson_title}.txt"
    filepath = os.path.join(course_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath


def create_course(topic):
    print(f"\n🧠 Generating course outline for: {topic}")
    outline_text = generate_course_outline(topic)
    if not outline_text:
        print("Failed to generate course outline. Exiting.")
        return

    # Save outline for reference/debugging
    course_dir = os.path.join("output", sanitize_filename(topic))
    os.makedirs(course_dir, exist_ok=True)
    outline_path = os.path.join(course_dir, "course_outline.txt")
    with open(outline_path, 'w', encoding='utf-8') as f:
        f.write(outline_text)

    print(f"DEBUG: Course outline saved to {outline_path}")

    outline = parse_outline(outline_text)
    if not outline:
        print("Parsed outline is empty. Please check the generated outline format.")
        return

    for ch_num, (chapter_key, details) in enumerate(outline.items(), start=1):
        print(f"\n📘 {chapter_key}: {details['title']}")
        for ls_num, lesson in enumerate(details["lessons"], start=1):
            print(f"  🎬 Lesson {ls_num}: {lesson}")
            script = generate_script(topic, details["title"], lesson)
            filepath = save_lesson(course_dir, ch_num, details["title"], ls_num, lesson, script)
            print(f"    Saved script to: {filepath}")

    print("\n✅ Course generation complete! Check the 'output' folder.")

import os

# # Replace this with your actual video creation function
# def create_video_from_script(script_text, output_path):
#     print(f"Creating video: {output_path}")
#     # Add your video creation logic here

def generate_videos_from_folder(course_folder, user_topic):
    import os
    import shutil
    import subprocess
    from pathlib import Path
    from pydub import AudioSegment
    import re

    # Setup output folder
    video_output_folder = os.path.join(course_folder, "videos")
    os.makedirs(video_output_folder, exist_ok=True)

    # Helper function to delete cache/log files
    def delete_path(path):
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.is_file():
                path.unlink()
            print(f"✅ Deleted: {path}")
        except Exception as e:
            print(f"❌ Error deleting {path}: {e}")

    cache_dirs = [Path.home() / "Library" / "Caches", Path("/Library/Caches")]
    log_dirs = [Path.home() / "Library" / "Logs", Path("/Library/Logs")]
    dirs_to_clean = cache_dirs + log_dirs

    print("🧹 Cleaning up system cache/log folders...")
    for d in dirs_to_clean:
        if d.exists():
            for item in d.iterdir():
                delete_path(item)
        else:
            print(f"⚠️ Directory not found: {d}")
    print("✅ Done cleaning up!")

    # Clear specific folders
    for folder_path in ["video_creation/", "audio/", "video_created/"]:
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    final_output = "final_video/myfinal.mp4"
    if os.path.exists(final_output):
        os.remove(final_output)

    # Process each script
    for filename in sorted(os.listdir(course_folder)):
        if filename.endswith(".txt") and filename != "course_outline.txt":
            filepath = os.path.join(course_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                script = f.read().strip()

            base_name = os.path.splitext(filename)[0]
            output_video_path = os.path.join(video_output_folder, f"{base_name}.mp4")
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"/Users/uday/Downloads/VIDEOYT/final_video/final_video_{timestamp}.mp4"
            output_file = create_video_from_script_with_whisperx(script, base_name, output_file)

            # Merge audio
            folder_path = "audio"
            combined = AudioSegment.empty()

            def extract_scene_number(filename):
                match = re.search(r"scene_(\d+)\.mp3", filename)
                return int(match.group(1)) if match else float('inf')

            mp3_files = sorted(
                [f for f in os.listdir(folder_path) if f.endswith(".mp3")],
                key=extract_scene_number
            )

            for file_name in mp3_files:
                file_path = os.path.join(folder_path, file_name)
                if os.path.getsize(file_path) == 0:
                    print(f"⚠️ Skipping empty file: {file_name}")
                    continue
                try:
                    audio = AudioSegment.from_mp3(file_path)
                    combined += audio
                except Exception as e:
                    print(f"❌ Could not decode {file_name}: {e}")
                    continue

            merged_audio_path = os.path.join(folder_path, "merged_audio.mp3")
            combined.export(merged_audio_path, format="mp3")
            print(f"✅ Merged audio saved to: {merged_audio_path}")

            # Convert to WAV
            wav_path = 'Wav2Lip/merged_audio.wav'
            mp3_to_wav(merged_audio_path, wav_path)

            import os
            import subprocess

            # Define all paths

            temp_result = "temp/result.avi"
            output_final = "video_created/output_with_audio.mp4"
            output_wav="/Users/uday/Downloads/VIDEOYT/Wav2Lip/merged_audio.wav"
            checkpoint_path = "checkpoints/wav2lip.pth"
            face_video = "Wav2Lip/output_video.mp4"


            run_wav2lip_inference(
                checkpoint_path="checkpoints/wav2lip.pth",
                face_video="Wav2Lip/output_video (4).mp4",
                audio_path="/Users/uday/Downloads/VIDEOYT/Wav2Lip/merged_audio.wav",
                output_video="video_created/output_with_audio1.mp4",
                static=True,
                fps=24,
                wav2lip_batch_size=256,
                resize_factor=4,
                out_height=360
            )



            # Step 2: Merge audio and video using FFmpeg
            print("🎵 Merging final video with audio...")
            subprocess.run([
                "ffmpeg", "-y",
                "-i", temp_result,
                "-i", output_wav,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "320k",         # Increased bitrate for better audio quality
                "-ar", "48000",         # Set sample rate to 48 kHz
                "-ac", "2",  
                "-shortest",
                output_final
            ])

            print(f"✅ Done! Final video with audio is saved to:\n{output_final}")

            import subprocess


  

            import subprocess

            command = [
                "ffmpeg", "-y",
                "-i", output_file,        # Background video (with or without audio)
                "-i", output_final,      # Avatar video (with audio)
                "-filter_complex",
                """
                [1:v]scale=444:667[avatar];
                [0:v]scale=1080:1920[bg];
                [bg][avatar]overlay=main_w-overlay_w-10:main_h-overlay_h-10[outv];
                [0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2[aout]
                """,
                "-map", "[outv]",
                "-map", "[aout]",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "48000",
                "-ac", "2",
                "-shortest",
                output_video_path
            ]






            print("▶️ Generating video with avatar in bottom-right corner...")
            subprocess.run(command)
            print(f"✅ Final video saved to: {output_video_path}")



            def save_file_topic_mappings(mappings, filename):
                """
                Save mappings of output file paths and topic names to a text file.
                
                Each line will be:
                output_file_path | topic_name
                
                :param mappings: List of tuples [(file_path1, topic1), (file_path2, topic2), ...]
                :param filename: Text file to save mappings
                """
                with open(filename, 'a') as f:
                    for file_path, topic in mappings:
                        f.write(f"{file_path} | {topic}\n")

            # Example usage:
            mappings = [
                (output_video_path, base_name)
                
            ]

            save_file_topic_mappings(mappings, "/Users/uday/Downloads/VIDEOYT/file_topic_map1.txt")


    print(f"✅ All videos saved in: {video_output_folder}")


if __name__ == "__main__":
    user_topic = input("Enter a topic for your video course: ").strip()

    


        # def get_urls_for_video(video_name, log_file="all_video_used_urls.txt"):
        #     urls = []
        #     with open(log_file, "r") as f:
        #         lines = f.readlines()

        #     collecting = False
        #     for line in lines:
        #         line = line.strip()
        #         if not line:
        #             collecting = False  # Stop when reaching a blank line
        #             continue
        #         if line == video_name:
        #             urls = []           # Clear in case video name appears multiple times
        #             collecting = True
        #             continue
        #         if collecting:
        #             urls.append(line)

        #     return urls


        # def generate_description_with_scene_links(base_description,feedback_link):
        #     description = sanitize_text(base_description)
        #     description = (
        #         f"{description}\n\n"
        #         f"📢 We'd love your feedback! Share your thoughts here 👉 {feedback_link}\n\n"
        #         "🎥 Scene Video Sources:\n"
        #     )
        #     urls= get_urls_for_video(output_file, log_file="all_video_used_urls.txt")
        #     for i, url in urls:
        #         description += f"Scene {i} video: {url}\n"

        #     description += (
        #         "\nCinematic Technology | Cybernetic Dreams by Alex-Productions\n"
        #         "https://youtu.be/NDYRjTti5Bw\n"
        #         "Music promoted by https://onsound.eu/\n"
        #     )

        #     return description


    if user_topic:
        create_course(user_topic)
        
        # Correct interpolation
        course_folder = os.path.join("output", sanitize_filename(user_topic))

        generate_videos_from_folder(course_folder, user_topic)
    else:
        print("No topic entered. Exiting.")


    


        # def get_urls_for_video(video_name, log_file="all_video_used_urls.txt"):
        #     urls = []
        #     with open(log_file, "r") as f:
        #         lines = f.readlines()

        #     collecting = False
        #     for line in lines:
        #         line = line.strip()
        #         if not line:
        #             collecting = False  # Stop when reaching a blank line
        #             continue
        #         if line == video_name:
        #             urls = []           # Clear in case video name appears multiple times
        #             collecting = True
        #             continue
        #         if collecting:
        #             urls.append(line)

        #     return urls


        # def generate_description_with_scene_links(base_description,feedback_link):
        #     description = sanitize_text(base_description)
        #     description = (
        #         f"{description}\n\n"
        #         f"📢 We'd love your feedback! Share your thoughts here 👉 {feedback_link}\n\n"
        #         "🎥 Scene Video Sources:\n"
        #     )
        #     urls= get_urls_for_video(output_file, log_file="all_video_used_urls.txt")
        #     for i, url in urls:
        #         description += f"Scene {i} video: {url}\n"

        #     description += (
        #         "\nCinematic Technology | Cybernetic Dreams by Alex-Productions\n"
        #         "https://youtu.be/NDYRjTti5Bw\n"
        #         "Music promoted by https://onsound.eu/\n"
        #     )

        #     return description

        

        # # Example Usage:
        # if output_file is not None:
        #     upload_video(output_file, user_topic)
        # else:
        #     print("Video creation failed.")

        # count -= 1
        # save_upload_status(count)
        # print(f"✅ Upload complete. Remaining uploads today: {count}")





