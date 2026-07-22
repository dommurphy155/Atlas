import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
ALLOWED_CHAT_IDS = [
    int(cid.strip())
    for cid in os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "").split(",")
    if cid.strip()
]

# Paths
SESSIONS_FILE = "/root/.claude/telegram_sessions.json"
CLAUDE_DIR = "/root/claude-code-haha"
MCP_CONFIG = os.path.join(CLAUDE_DIR, ".mcp.json")
SKILLS_DIR = os.path.join(CLAUDE_DIR, "src/skills/bundled")

# Ensure sessions directory exists
os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)


import logging
import logging.handlers

# Setup logging
LOG_FILE = "/var/log/telegram-claude-bot.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Root logger setup
logger = logging.getLogger("telegram_bot")
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# File handler with rotation (10MB per file, 5 backups)
file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=10*1024*1024, backupCount=5
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def log(message: str, level: str = "INFO"):
    """Log a message with the specified level."""
    level = level.upper()
    if level == "DEBUG":
        logger.debug(message)
    elif level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "CRITICAL":
        logger.critical(message)
    else:
        logger.info(message)


def load_sessions() -> dict:
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log(f"Error loading sessions: {e}")
    return {}


def save_sessions(sessions: dict):
    try:
        with open(SESSIONS_FILE, "w") as f:
            json.dump(sessions, f, indent=2)
    except IOError as e:
        log(f"Error saving sessions: {e}")


def escape_markdown_v2(text: str) -> str:
    chars = r"_\*\[\]\(\)~`>#+\-=|{}\.!"
    return re.sub(f"([{re.escape(chars)}])", r"\\\1", text)


def format_for_telegram(text: str) -> str:
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text.strip()


def escape_telegram_markdown(text: str) -> str:
    ESCAPE_CHARS = r'_*[]()~`>#+-=|{}.!'

    def escape_content(s: str) -> str:
        result = []
        for c in s:
            if c in ESCAPE_CHARS:
                result.append(f'\\{c}')
            else:
                result.append(c)
        return ''.join(result)

    result = []
    i = 0
    n = len(text)

    while i < n:
        if text[i:i+3] == '```':
            end = text.find('```', i+3)
            if end != -1:
                content = text[i+3:end]
                result.append(f'```{content}```')
                i = end + 3
                continue
        elif text[i:i+2] == '**':
            end = text.find('**', i+2)
            if end != -1:
                content = text[i+2:end]
                escaped = escape_content(content)
                result.append(f'*{escaped}*')
                i = end + 2
                continue
        elif text[i] == '*':
            end = text.find('*', i+1)
            if end != -1:
                content = text[i+1:end]
                escaped = escape_content(content)
                result.append(f'*{escaped}*')
                i = end + 1
                continue
        elif text[i] == '_':
            end = text.find('_', i+1)
            if end != -1:
                content = text[i+1:end]
                escaped = escape_content(content)
                result.append(f'_{escaped}_')
                i = end + 1
                continue
        elif text[i] == '`':
            end = text.find('`', i+1)
            if end != -1:
                content = text[i+1:end]
                content = content.replace('\\', '\\\\').replace('`', '\\`')
                result.append(f'`{content}`')
                i = end + 1
                continue
        elif text[i] == '[':
            close_bracket = text.find(']', i+1)
            if close_bracket != -1 and close_bracket + 1 < n and text[close_bracket + 1] == '(':
                close_paren = text.find(')', close_bracket + 2)
                if close_paren != -1:
                    link_text = text[i+1:close_bracket]
                    url = text[close_bracket+2:close_paren]
                    url = url.replace(')', r'\)').replace('(', r'\(')
                    result.append(f'[{link_text}]({url})')
                    i = close_paren + 1
                    continue

        char = text[i]
        if char in ESCAPE_CHARS:
            result.append(f'\\{char}')
        else:
            result.append(char)
        i += 1

    return ''.join(result)


def truncate_for_telegram(text: str, max_length: int = 4000) -> list:
    if len(text) <= max_length:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        cutoff = text.rfind("\n", 0, max_length)
        if cutoff == -1:
            cutoff = text.rfind(" ", 0, max_length)
        if cutoff == -1:
            cutoff = max_length
        chunks.append(text[:cutoff])
        text = text[cutoff:].lstrip()
    return chunks


class MediaProcessor:
    """Handle media processing using the media skill."""

    def __init__(self):
        self.temp_dir = "/tmp/telegram_bot_media"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.nvidia_api_key = os.environ.get("NVIDIA_API_KEY", "")
        self.nvidia_image_key = os.environ.get("NVIDIA_IMAGE_API_KEY", "")
        self._check_deps()

    def _check_deps(self):
        """Check dependencies at startup."""
        if not self.nvidia_api_key:
            log("⚠️  NVIDIA_API_KEY not set - voice and vision features disabled")
        if not self.nvidia_image_key:
            log("⚠️  NVIDIA_IMAGE_API_KEY not set - image generation disabled")

    def download_file(self, url: str) -> Optional[str]:
        """Download file from URL to temp directory."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            filename = os.path.join(self.temp_dir, f"dl_{datetime.now().timestamp()}")
            with open(filename, "wb") as f:
                f.write(response.content)
            return filename
        except Exception as e:
            log(f"Error downloading file: {e}")
            return None

    def speech_to_text(self, audio_path: str, language: str = 'en') -> str:
        """Transcribe audio to text using NVIDIA's Canary model via NIM API."""
        if not self.nvidia_api_key:
            return "[NVIDIA_API_KEY not configured - can't transcribe voice]"

        try:
            import subprocess
            import tempfile

            # Convert to WAV if needed
            wav_path = audio_path
            if not audio_path.lower().endswith('.wav'):
                wav_fd, wav_path = tempfile.mkstemp(suffix='.wav')
                os.close(wav_fd)
                try:
                    subprocess.run(
                        ['ffmpeg', '-i', audio_path, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', wav_path, '-y'],
                        check=True, capture_output=True, timeout=30
                    )
                except Exception as e:
                    log(f"FFmpeg conversion error: {e}")
                    os.unlink(wav_path)
                    return "[Audio conversion failed]"

            # Read audio file
            with open(wav_path, 'rb') as f:
                audio_buffer = f.read()

            # Clean up temp file if we created one
            if wav_path != audio_path:
                os.unlink(wav_path)

            # Call NVIDIA NIM ASR API (direct inference endpoint)
            asr_url = 'https://integrate.api.nvidia.com/v1/audio/transcriptions'
            files = {'file': ('audio.wav', audio_buffer, 'audio/wav')}
            data = {'model': 'nvidia/canary-1b', 'language': language}
            headers = {'Authorization': f'Bearer {self.nvidia_api_key}'}

            log(f"Calling ASR API with {len(audio_buffer)} bytes")
            response = requests.post(asr_url, headers=headers, files=files, data=data, timeout=60)
            log(f"ASR response: {response.status_code}")


            response.raise_for_status()
            result = response.json()

            # Handle different response formats
            if 'text' in result:
                return result['text'].strip()
            elif 'transcription' in result:
                return result['transcription'].strip()
            elif 'result' in result:
                return result['result'].strip()
            else:
                return f"[Unexpected response: {list(result.keys())}]"

        except Exception as e:
            log(f"STT error: {e}")
            return "[Audio transcription failed]"

    def text_to_speech(self, text: str, output_path: Optional[str] = None, voice: str = 'en-GB-RyanNeural') -> Optional[str]:
        """Convert text to speech using Edge TTS and convert to OGG for Telegram."""
        if not output_path:
            output_path = os.path.join(self.temp_dir, f"tts_{datetime.now().timestamp()}.mp3")
        ogg_path = output_path.replace('.mp3', '.ogg')
        try:
            import subprocess
            # Generate MP3 with edge-tts
            cmd = [
                'edge-tts',
                '--voice', voice,
                '--rate', '+2%',
                '--pitch', '-2Hz',
                '--text', text,
                '--write-media', output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
            if not os.path.exists(output_path):
                return None
            # Convert MP3 to OGG for Telegram voice messages
            subprocess.run(
                ['ffmpeg', '-i', output_path, '-c:a', 'libopus', '-b:a', '24k', ogg_path, '-y'],
                check=True, capture_output=True, timeout=30
            )
            # Clean up MP3
            os.remove(output_path)
            return ogg_path if os.path.exists(ogg_path) else None
        except Exception as e:
            log(f"TTS error: {e}")
            return None

    def analyze_image(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Analyze image using Kimi-K2.5 vision model."""
        if not self.nvidia_api_key:
            return "[NVIDIA_API_KEY not configured - can't analyze images]"

        try:
            # Read image file
            with open(image_path, 'rb') as f:
                image_buffer = f.read()

            # Determine mime type from extension
            ext = image_path.split('.')[-1].lower() if '.' in image_path else 'png'
            mime_type = 'image/jpeg' if ext in ['jpg', 'jpeg'] else 'image/webp' if ext == 'webp' else 'image/png'

            base64_image = image_buffer.hex()
            # Actually use base64 encoding properly
            import base64
            base64_image = base64.b64encode(image_buffer).decode('utf-8')

            llm_url = 'https://integrate.api.nvidia.com/v1/chat/completions'
            payload = {
                'model': 'google/gemma-4-31b-it',
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': prompt or "Describe this image in detail"},
                            {'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{base64_image}'}}
                        ]
                    }
                ],
                'max_tokens': 1024,
                'temperature': 0.7
            }

            headers = {'Authorization': f'Bearer {self.nvidia_api_key}', 'Content-Type': 'application/json'}
            response = requests.post(llm_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            if not isinstance(result, dict):
                return "[Unexpected response format]"
            choices = result.get('choices', [])
            if not isinstance(choices, list) or not choices:
                return "[No analysis available]"
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            message = first_choice.get('message', {}) if isinstance(first_choice.get('message'), dict) else {}
            return message.get('content', '').strip() or "[No analysis available]"

        except Exception as e:
            log(f"Vision error: {e}")
            return "[Image analysis failed]"

    def generate_image(self, prompt: str, output_path: Optional[str] = None, aspect_ratio: str = "16:9") -> Optional[str]:
        """Generate image using NVIDIA Stable Diffusion 3."""
        if not self.nvidia_image_key:
            log("NVIDIA_IMAGE_API_KEY not set - can't generate images")
            return None

        if not output_path:
            output_path = os.path.join(self.temp_dir, f"gen_{datetime.now().timestamp()}.png")
        try:
            import base64

            image_url = 'https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium'
            payload = {
                'prompt': prompt,
                'cfg_scale': 5,
                'aspect_ratio': aspect_ratio,
                'seed': 0,
                'steps': 50,
                'negative_prompt': ''
            }

            headers = {
                'Authorization': f'Bearer {self.nvidia_image_key}',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }

            response = requests.post(image_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            if not isinstance(data, dict):
                return None

            # Extract base64 image
            image_base64 = None
            artifacts = data.get('artifacts', [])
            if isinstance(artifacts, list) and artifacts and isinstance(artifacts[0], dict) and artifacts[0].get('base64'):
                image_base64 = artifacts[0]['base64']
            elif data.get('image'):
                image_base64 = data['image']
            elif isinstance(data.get('images'), list) and data['images']:
                image_base64 = data['images'][0]

            if image_base64:
                with open(output_path, 'wb') as f:
                    f.write(base64.b64decode(image_base64))
                return output_path

            return None
        except Exception as e:
            log(f"Image generation error: {e}")
            return None

    def cleanup(self, filepath: str):
        """Remove temp file."""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass


class TelegramAPI:
    def __init__(self, token: str):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}"

    def _request(self, method: str, **params) -> dict:
        url = f"{self.base_url}/{method}"
        try:
            response = requests.post(url, json=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                log(f"API returned non-dict: {type(data)}")
                return {}
            if not data.get("ok"):
                log(f"API error in {method}: {data.get('description')}")
                return {}
            return data
        except requests.RequestException as e:
            log(f"Request error in {method}: {e}")
            # Log the params that caused the error (but truncate text for readability)
            debug_params = {k: (v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v) for k, v in params.items()}
            log(f"Failed params: {debug_params}")
            return {}

    def get_updates(self, offset: int = 0, limit: int = 100) -> list:
        url = f"{self.base_url}/getUpdates"
        try:
            response = requests.post(
                url,
                json={"offset": offset, "limit": limit, "timeout": 60, "allowed_updates": ["message", "callback_query"]},
                timeout=70,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                log(f"getUpdates returned non-dict: {type(data)}")
                return []
            if not data.get("ok"):
                log(f"API error: {data.get('description')}")
                return []
            result = data.get("result", [])
            return result if isinstance(result, list) else []
        except requests.Timeout:
            return []
        except requests.RequestException as e:
            log(f"Request error: {e}")
            return []

    def send_message(self, chat_id: int, text: str, parse_mode: Optional[str] = None, reply_markup: Optional[dict] = None) -> dict:
        params = {"chat_id": chat_id, "text": text}
        if parse_mode:
            params["parse_mode"] = parse_mode
        if reply_markup:
            params["reply_markup"] = json.dumps(reply_markup)
        # Retry once on rate limit
        for attempt in range(3):
            url = f"{self.base_url}/sendMessage"
            try:
                response = requests.post(url, json=params, timeout=30)
                if response.status_code == 429:
                    retry_after = response.json().get("parameters", {}).get("retry_after", 10)
                    log(f"Rate limited on sendMessage, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    return {}
                return data
            except requests.RequestException as e:
                log(f"Request error in sendMessage: {e}")
                if attempt < 2:
                    time.sleep(3)
        return {}

    def edit_message(self, chat_id: int, message_id: int, text: str, parse_mode: Optional[str] = None) -> dict:
        params = {"chat_id": chat_id, "message_id": message_id, "text": text[:4000]}
        if parse_mode:
            params["parse_mode"] = parse_mode
        return self._request("editMessageText", **params)

    def send_chat_action(self, chat_id: int, action: str = "typing"):
        self._request("sendChatAction", chat_id=chat_id, action=action)

    def answer_callback(self, callback_query_id: str):
        self._request("answerCallbackQuery", callback_query_id=callback_query_id)

    def get_file(self, file_id: str) -> Optional[str]:
        """Get file download URL from Telegram."""
        result = self._request("getFile", file_id=file_id)
        if result.get("ok"):
            file_path = result["result"].get("file_path")
            if file_path:
                return f"https://api.telegram.org/file/bot{self.token}/{file_path}"
        return None

    def send_voice(self, chat_id: int, audio_path: str, caption: Optional[str] = None) -> dict:
        """Send voice message (OGG format)."""
        url = f"{self.base_url}/sendVoice"
        try:
            with open(audio_path, "rb") as f:
                files = {"voice": f}
                data = {"chat_id": chat_id}
                if caption:
                    data["caption"] = caption[:1024]
                response = requests.post(url, data=data, files=files, timeout=60)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            log(f"Error sending voice: {e}")
            return {}

    def send_photo(self, chat_id: int, photo_path: str, caption: Optional[str] = None) -> dict:
        """Send photo."""
        url = f"{self.base_url}/sendPhoto"
        try:
            with open(photo_path, "rb") as f:
                files = {"photo": f}
                data = {"chat_id": chat_id}
                if caption:
                    data["caption"] = caption[:1024]
                response = requests.post(url, data=data, files=files, timeout=60)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            log(f"Error sending photo: {e}")
            return {}


class ClaudeBot:
    def __init__(self):
        if not BOT_TOKEN:
            log("Error: TELEGRAM_BOT_TOKEN not set")
            sys.exit(1)
        self.api = TelegramAPI(BOT_TOKEN)
        self.sessions = load_sessions()
        self.offset = 0
        self.media = MediaProcessor()
        self._processed_messages = set()  # Track processed message IDs to prevent duplicates
        self._max_tracked = 1000  # Limit memory usage
        self._current_process = None  # Track current Claude subprocess
        self._stop_requested = False  # Flag to signal stop
        # Job tracking for concurrent requests and /stop
        import threading
        self._jobs = {}  # job_id -> {"thread": t, "stop_event": e, "chat_id": cid, "preview": str, "message_id": mid}
        self._jobs_lock = threading.Lock()
        self._job_counter = 0

    def is_allowed(self, chat_id: int) -> bool:
        if not ALLOWED_CHAT_IDS:
            return True
        return chat_id in ALLOWED_CHAT_IDS

    def get_chat_session(self, chat_id: int) -> Optional[str]:
        chat_data = self.sessions.get(str(chat_id), {})
        return chat_data.get("active_session")

    def create_session(self, chat_id: int, session_id: str, first_message: str):
        title = first_message[:50] + "..." if len(first_message) > 50 else first_message
        chat_data = self.sessions.setdefault(str(chat_id), {"sessions": []})
        chat_data["active_session"] = session_id
        session_info = {
            "id": session_id,
            "title": title,
            "created": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
        }
        chat_data["sessions"].append(session_info)
        save_sessions(self.sessions)
        log(f"Created session {session_id[:8]}... for chat {chat_id}")

    def update_session_id(self, chat_id: int, old_session_id: Optional[str], new_session_id: str, first_message: str):
        if old_session_id and old_session_id == new_session_id:
            return
        chat_data = self.sessions.setdefault(str(chat_id), {"sessions": []})
        if old_session_id:
            sessions = chat_data.get("sessions", [])
            chat_data["sessions"] = [s for s in sessions if s["id"] != old_session_id]
        existing = None
        for s in chat_data.get("sessions", []):
            if s["id"] == new_session_id:
                existing = s
                break
        if not existing:
            title = first_message[:50] + "..." if len(first_message) > 50 else first_message
            session_info = {
                "id": new_session_id,
                "title": title,
                "created": datetime.now().isoformat(),
                "last_used": datetime.now().isoformat(),
            }
            chat_data["sessions"].append(session_info)
        chat_data["active_session"] = new_session_id
        save_sessions(self.sessions)
        log(f"Set session {new_session_id[:8]}... for chat {chat_id}")

    def update_session_timestamp(self, chat_id: int, session_id: str):
        chat_data = self.sessions.get(str(chat_id), {})
        for session in chat_data.get("sessions", []):
            if session["id"] == session_id:
                session["last_used"] = datetime.now().isoformat()
                break
        save_sessions(self.sessions)

    def clear_active_session(self, chat_id: int):
        chat_data = self.sessions.get(str(chat_id), {})
        if "active_session" in chat_data:
            del chat_data["active_session"]
            save_sessions(self.sessions)
            log(f"Cleared active session for chat {chat_id}")

    def delete_session(self, chat_id: int, session_id: str) -> bool:
        chat_data = self.sessions.get(str(chat_id), {})
        sessions = chat_data.get("sessions", [])
        for i, session in enumerate(sessions):
            if session["id"] == session_id:
                sessions.pop(i)
                if chat_data.get("active_session") == session_id:
                    chat_data["active_session"] = None
                save_sessions(self.sessions)
                log(f"Deleted session {session_id[:8]}... for chat {chat_id}")
                return True
        return False

    def set_active_session(self, chat_id: int, session_id: str) -> bool:
        chat_data = self.sessions.get(str(chat_id), {})
        for session in chat_data.get("sessions", []):
            if session["id"] == session_id:
                chat_data["active_session"] = session_id
                save_sessions(self.sessions)
                log(f"Set active session {session_id[:8]}... for chat {chat_id}")
                return True
        return False

    def call_claude(self, message: str, session_id: Optional[str] = None) -> tuple:
        cmd = [
            "claude", "-p", "--permission-mode", "bypassPermissions",
            "--output-format", "json",
        ]
        if session_id:
            cmd.extend(["--resume", session_id])
        cmd.append(message)
        log(f"Calling Claude with session={session_id[:8] if session_id else 'new'}...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=CLAUDE_DIR, timeout=2700)
            if result.returncode != 0:
                log(f"Claude error: {result.stderr}")
                return f"Error: Claude returned exit code {result.returncode}", session_id
            try:
                # Claude may output internal protocol mixed with JSON - extract just the JSON result line
                stdout = result.stdout.strip()
                # Try to find a line that's valid JSON with a "result" field
                for line in stdout.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict) and "result" in data:
                            returned_session_id = data.get("session_id", session_id)
                            return data.get("result", "No result in response"), returned_session_id
                    except json.JSONDecodeError:
                        continue
                # Fallback: try parsing entire stdout as JSON
                data = json.loads(stdout)
                if isinstance(data, dict):
                    returned_session_id = data.get("session_id", session_id)
                    return data.get("result", "No result in response"), returned_session_id
                else:
                    return str(data), session_id
            except json.JSONDecodeError:
                # Not JSON at all, return raw stdout
                return stdout, session_id
        except subprocess.TimeoutExpired:
            log("Claude timeout")
            return "Error: Claude timed out after 5 minutes", session_id
        except Exception as e:
            log(f"Claude exception: {e}")
            return f"Error calling Claude: {e}", session_id

    def call_claude_with_streaming(self, chat_id: int, message: str, session_id: Optional[str] = None) -> tuple:
        import threading
        import queue

        cmd = [
            "claude", "-p", "--permission-mode", "bypassPermissions",
            "--output-format", "json",
        ]
        if session_id:
            cmd.extend(["--resume", session_id])
        cmd.append(message)

        log(f"Calling Claude with streaming, session={session_id[:8] if session_id else 'new'}...")

        msg_data = self.api.send_message(chat_id, "⏳ Thinking...")
        result_data = msg_data.get("result") if isinstance(msg_data, dict) else {}
        message_id = result_data.get("message_id") if isinstance(result_data, dict) else None

        if not message_id:
            return self.call_claude(message, session_id)

        output_queue = queue.Queue()
        stop_event = threading.Event()

        def run_claude():
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=CLAUDE_DIR)
                # Poll so we can honour stop_event
                while proc.poll() is None:
                    if stop_event.is_set():
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except Exception:
                            proc.kill()
                        output_queue.put(("stopped", "", None))
                        return
                    import time as _time
                    _time.sleep(0.2)
                stdout, stderr = proc.communicate()
                if stop_event.is_set():
                    output_queue.put(("stopped", "", None))
                    return
                if proc.returncode == 0:
                    try:
                        text = stdout.strip()
                        for line in text.split("\n"):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                if isinstance(data, dict) and "result" in data:
                                    output_queue.put(("success", data.get("result", ""), data.get("session_id")))
                                    return
                            except json.JSONDecodeError:
                                continue
                        data = json.loads(text)
                        if isinstance(data, dict):
                            output_queue.put(("success", data.get("result", ""), data.get("session_id")))
                        else:
                            output_queue.put(("success", str(data), None))
                    except json.JSONDecodeError:
                        output_queue.put(("success", stdout, None))
                else:
                    output_queue.put(("error", f"Exit code {proc.returncode}: {stderr}", None))
            except Exception as e:
                output_queue.put(("error", str(e), None))

        thread = threading.Thread(target=run_claude, daemon=True)
        thread.start()
        jid = self._register_job(chat_id, message, stop_event, thread, message_id)

        dots = ["⏳", "⌛", "⏳", "⌛"]
        i = 0
        status = None
        spinner_interval = 8  # update every 8 seconds to avoid rate limits
        spinner_tick = 0
        spinner_max_updates = 37  # stop after ~5 minutes of spinner updates
        while thread.is_alive():
            try:
                status = output_queue.get(timeout=1)
                break
            except queue.Empty:
                spinner_tick += 1
                if spinner_tick >= spinner_interval and i < spinner_max_updates:
                    spinner_tick = 0
                    self.api.edit_message(chat_id, message_id, f"{dots[i % 4]} Thinking...")
                    i += 1
        if status is None:
            try:
                status = output_queue.get(timeout=2)
            except queue.Empty:
                status = ("error", "No output", None)

        thread.join(timeout=2)
        self._unregister_job(jid)

        if status[0] == "stopped":
            self.api.edit_message(chat_id, message_id, "🛑 Request cancelled.")
            return "", session_id

        if status[0] == "error":
            self.api.edit_message(chat_id, message_id, f"❌ Error: {escape_telegram_markdown(str(status[1]))[:1000]}")
            return status[1], session_id

        response = status[1]
        returned_session_id = status[2] if len(status) > 2 else session_id
        full_response = response.strip() if response else "No response"

        if not full_response or full_response == "No response":
            self.api.edit_message(chat_id, message_id, "🤔 No response from Claude")
            return "No response", returned_session_id

        if full_response.startswith("IMAGE:"):
            self.api._request("deleteMessage", chat_id=chat_id, message_id=message_id)
            self.route_response(chat_id, full_response)
            return full_response, returned_session_id

        if full_response.startswith("VOICE:"):
            self.api._request("deleteMessage", chat_id=chat_id, message_id=message_id)
            self.route_response(chat_id, full_response)
            return full_response, returned_session_id

        # Send potentially long response in chunks
        self._send_long_response(chat_id, full_response, edit_message_id=message_id)
        return full_response, returned_session_id


    def handle_start(self, chat_id: int):
        welcome = """👋 *Welcome to Claude Code Bot!*

I connect you to Claude Code CLI through Telegram.

*Commands:*
• Send any message to chat with Claude
• `/new` - Start a fresh conversation
• `/sessions` - Manage your conversation history
• `/skills` - List available skills and MCP servers

*Media Features:*
• 🎤 Send voice messages - I'll transcribe and reply
• 📸 Send photos - I'll analyze and describe them
• I can reply with voice or generate images when it makes sense

Your conversations are persisted and can be resumed anytime.
"""
        self.api.send_message(chat_id, welcome, parse_mode="Markdown")

    def handle_new(self, chat_id: int):
        self.clear_active_session(chat_id)
        self.api.send_message(
            chat_id,
            "✅ *New session started*\n\nYour next message will begin a fresh conversation with Claude.",
            parse_mode="Markdown",
        )

    def get_session_context(self, session_id: str) -> str:
        try:
            cmd = [
                "claude", "-p", "--permission-mode", "bypassPermissions",
                "--output-format", "json",
                "--resume", session_id,
                "In 2 sentences, summarize what we were just working on.",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=CLAUDE_DIR, timeout=60)
            if result.returncode == 0:
                stdout = result.stdout.strip()
                # Try to find a line that's valid JSON with a "result" field
                for line in stdout.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict) and "result" in data:
                            return data.get("result", "No context available")
                    except json.JSONDecodeError:
                        continue
                # Fallback: try parsing entire stdout
                data = json.loads(stdout)
                if isinstance(data, dict):
                    return data.get("result", "No context available")
                return str(data)[:200]
            return "Could not retrieve context"
        except Exception as e:
            log(f"Error getting context: {e}")
            return "Error retrieving context"

    def rename_session(self, chat_id: int, session_id: str, new_title: str):
        chat_data = self.sessions.get(str(chat_id), {})
        for session in chat_data.get("sessions", []):
            if session["id"] == session_id:
                session["title"] = new_title
                save_sessions(self.sessions)
                return True
        return False

    def session_exists(self, chat_id: int, session_id: str) -> bool:
        chat_data = self.sessions.get(str(chat_id), {})
        for session in chat_data.get("sessions", []):
            if session["id"] == session_id:
                return True
        return False

    def validate_session(self, session_id: str) -> tuple:
        """Validate a session by attempting to ping it via Claude CLI."""
        cmd = [
            "claude", "-p", "--dangerously-skip-permissions",
            "--output-format", "json",
            "--resume", session_id,
            "hi reply 1 word 'hello'",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=CLAUDE_DIR, timeout=30)
            # If exit code is 0 and we got JSON back with a result, session is valid
            if result.returncode == 0 and result.stdout.strip():
                stdout = result.stdout.strip()
                # Try to find a line that's valid JSON with a "result" field
                for line in stdout.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict) and "result" in data:
                            return True, data.get("session_id", session_id)
                    except json.JSONDecodeError:
                        continue
                # Fallback: try parsing entire stdout
                try:
                    data = json.loads(stdout)
                    if isinstance(data, dict) and "result" in data:
                        return True, data.get("session_id", session_id)
                except json.JSONDecodeError:
                    pass
            return False, None
        except Exception as e:
            log(f"Error validating session: {e}")
            return False, None

    def handle_find_session_result(self, chat_id: int, session_id: str):
        chat_data = self.sessions.setdefault(str(chat_id), {"sessions": []})

        # Clear the pending flag
        if "find_pending" in chat_data:
            del chat_data["find_pending"]

        # Check if session already exists
        if self.session_exists(chat_id, session_id):
            keyboard = {
                "inline_keyboard": [
                    [
                        {"text": "✅ Yes", "callback_data": f"find_continue:{session_id}"},
                        {"text": "❌ No", "callback_data": "find_cancel"},
                    ]
                ]
            }
            self.api.send_message(
                chat_id,
                "✅ *Session found*\n\nWould you like to continue this thread now?",
                parse_mode="MarkdownV2",
                reply_markup=keyboard,
            )
            save_sessions(self.sessions)
            return

        # Validate the session
        self.api.send_chat_action(chat_id, "typing")
        is_valid, validated_id = self.validate_session(session_id)

        if not is_valid:
            keyboard = {
                "inline_keyboard": [
                    [
                        {"text": "🔄 Try again", "callback_data": "find_session"},
                        {"text": "❌ Exit", "callback_data": "find_cancel"},
                    ]
                ]
            }
            self.api.send_message(
                chat_id,
                "❌ *Session not found*\n\nCouldn't find that session ID\\. Please ensure it's correct and try again\\.",
                parse_mode="MarkdownV2",
                reply_markup=keyboard,
            )
            save_sessions(self.sessions)
            return

        # Add the session
        session_info = {
            "id": validated_id or session_id,
            "title": "Found Session",
            "created": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
        }
        chat_data["sessions"].append(session_info)
        save_sessions(self.sessions)

        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "✅ Yes", "callback_data": f"find_continue:{validated_id or session_id}"},
                    {"text": "❌ No", "callback_data": "find_cancel"},
                ]
            ]
        }
        self.api.send_message(
            chat_id,
            "✅ *Session found*\n\nWould you like to continue this thread now?",
            parse_mode="MarkdownV2",
            reply_markup=keyboard,
        )

    def get_session_by_index(self, chat_id: int, index: int) -> Optional[dict]:
        chat_data = self.sessions.get(str(chat_id), {})
        sessions = chat_data.get("sessions", [])
        if 0 <= index < len(sessions):
            return sessions[index]
        return None

    def build_session_message(self, chat_id: int, page: int) -> tuple:
        chat_data = self.sessions.get(str(chat_id), {})
        sessions = chat_data.get("sessions", [])
        # Reverse sessions: newest first
        sessions = list(reversed(sessions))
        total = len(sessions)

        if not sessions:
            return (
                "📭 *No sessions yet*\n\nSend a message to start your first conversation\\!",
                None
            )

        page = max(0, min(page, total - 1))

        session = sessions[page]
        sid = session["id"]
        title = session.get("title", "Untitled")

        context_data = chat_data.get("session_context", {})
        context_line = ""
        if sid in context_data:
            ctx = escape_telegram_markdown(context_data[sid][:300])
            context_line = f"\n🧠 {ctx}\n"
        else:
            context_line = "\n🧠 _Press 📖 Context to load context_\n"

        text = (
            "📚 *Your Sessions*\n"
            "Manage and navigate saved sessions\n\n"
            f"📚 Session {page + 1} / {total}\n\n"
            f"🤖 {escape_telegram_markdown(title)}\n"
            f"ID: `{sid}`"
            f"{context_line}"
        )

        # Build navigation row - conditionally show buttons based on position
        nav_row = []

        # ⏮️ Back 5 - only show if we're past page 1
        if page >= 2:
            nav_row.append({"text": "⏮️", "callback_data": f"nav:{max(0, page - 5)}"})
        # ⬅️ Back 1 - at page 0, goes to last session (circular). Otherwise goes back 1.
        if page == 0:
            nav_row.append({"text": "⬅️", "callback_data": f"nav:{total - 1}"})
        else:
            nav_row.append({"text": "⬅️", "callback_data": f"nav:{page - 1}"})

        # ➡️ Forward 1 - wraps to first session if at end
        if page == total - 1:
            nav_row.append({"text": "➡️", "callback_data": "nav:0"})
        else:
            nav_row.append({"text": "➡️", "callback_data": f"nav:{page + 1}"})

        # ⏭️ Forward 5 - jumps to last if near end
        if page >= total - 5:
            nav_row.append({"text": "⏭️", "callback_data": f"nav:{total - 1}"})
        else:
            nav_row.append({"text": "⏭️", "callback_data": f"nav:{min(total - 1, page + 5)}"})

        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "▶️ Resume",  "callback_data": f"continue:{sid}"},
                    {"text": "📖 Context", "callback_data": f"context:{sid}:{page}"},
                ],
                [
                    {"text": "📝 Rename",  "callback_data": f"rename:{sid}"},
                    {"text": "🗑️ Delete",  "callback_data": f"delete:{sid}:{page}"},
                ],
                [
                    {"text": "🔎 Find session", "callback_data": "find_session"},
                ],
                nav_row,
            ]
        }

        return text, keyboard

    def handle_sessions(self, chat_id: int):
        chat_data = self.sessions.setdefault(str(chat_id), {"sessions": []})
        chat_data["sessions_page"] = 0
        save_sessions(self.sessions)

        text, keyboard = self.build_session_message(chat_id, 0)
        self.api.send_message(
            chat_id,
            text,
            parse_mode="MarkdownV2",
            reply_markup=keyboard if keyboard else None,
        )

    def handle_skills(self, chat_id: int):
        sections = ["🛠️ *Available Skills & MCP Servers*\n"]
        sections.append("\n*📡 MCP Servers:*")
        if os.path.exists(MCP_CONFIG):
            try:
                with open(MCP_CONFIG) as f:
                    mcp_data = json.load(f)
                servers = mcp_data.get("mcpServers", {})
                if servers:
                    for name, config in servers.items():
                        sections.append(f"• `{name}` - {config.get('command', 'unknown')}")
                else:
                    sections.append("_No MCP servers configured_")
            except Exception as e:
                sections.append(f"_Error reading MCP config: {e}_")
        else:
            sections.append("_No .mcp.json found_")

        sections.append("\n*📦 Bundled Skills:*")
        if os.path.exists(SKILLS_DIR):
            try:
                skills = [
                    f.replace(".js", "").replace(".ts", "")
                    for f in os.listdir(SKILLS_DIR)
                    if f.endswith((".js", ".ts")) and not f.startswith("index")
                ]
                for skill in sorted(set(skills)):
                    sections.append(f"• `{skill}`")
            except Exception as e:
                sections.append(f"_Error reading skills: {e}_")
        else:
            sections.append("_Skills directory not found_")

        text = "\n".join(sections)
        if len(text) > 4000:
            parts = truncate_for_telegram(text)
            for part in parts:
                self.api.send_message(chat_id, part, parse_mode="Markdown")
        else:
            self.api.send_message(chat_id, text, parse_mode="Markdown")

    def handle_callback(self, callback: dict):
        query_id = callback.get("id") if isinstance(callback, dict) else None
        data = callback.get("data", "") if isinstance(callback, dict) else ""
        message = callback.get("message") if isinstance(callback, dict) else None
        chat_id = message.get("chat", {}).get("id") if isinstance(message, dict) else None
        message_id = message.get("message_id") if isinstance(message, dict) else None

        self.api.answer_callback(query_id)

        if data == "noop":
            return

        if data.startswith("continue:"):
            session_id = data.split(":", 1)[1]
            if self.set_active_session(chat_id, session_id):
                self.api.send_message(
                    chat_id,
                    "✅ *Session resumed*\n\nContinuing conversation\\.\\.\\.",
                    parse_mode="MarkdownV2",
                )
            else:
                self.api.send_message(chat_id, "❌ Session not found")

        elif data.startswith("delete:"):
            _, session_id, page_str = data.split(":", 2)
            page = int(page_str)
            if self.delete_session(chat_id, session_id):
                chat_data = self.sessions.get(str(chat_id), {})
                total = len(chat_data.get("sessions", []))
                new_page = min(page, total - 1) if total > 0 else 0
                chat_data["sessions_page"] = new_page
                save_sessions(self.sessions)
                text, keyboard = self.build_session_message(chat_id, new_page)
                self.api.edit_message(chat_id, message_id, text, parse_mode="MarkdownV2")
                if keyboard:
                    self.api._request(
                        "editMessageReplyMarkup",
                        chat_id=chat_id,
                        message_id=message_id,
                        reply_markup=json.dumps(keyboard),
                    )
            else:
                self.api.send_message(chat_id, "❌ Session not found")

        elif data.startswith("context:"):
            _, session_id, page_str = data.split(":", 2)
            page = int(page_str)

            # Animated loading with timer
            dots = ["⏳", "⌛", "⏳", "⌛"]
            for i in range(6):  # Show animation for ~3 seconds
                loading_text = (
                    "📚 *Your Sessions*\n"
                    "Manage and navigate saved sessions\n\n"
                    f"{dots[i % 4]} Loading Session Context\\.\\.\\."
                )
                self.api.edit_message(chat_id, message_id, loading_text, parse_mode="MarkdownV2")
                time.sleep(0.5)

            context = self.get_session_context(session_id)
            chat_data = self.sessions.setdefault(str(chat_id), {"sessions": []})
            chat_data.setdefault("session_context", {})[session_id] = context
            chat_data["sessions_page"] = page
            save_sessions(self.sessions)
            text, keyboard = self.build_session_message(chat_id, page)
            self.api.edit_message(chat_id, message_id, text, parse_mode="MarkdownV2")
            if keyboard:
                self.api._request(
                    "editMessageReplyMarkup",
                    chat_id=chat_id,
                    message_id=message_id,
                    reply_markup=json.dumps(keyboard),
                )

        elif data.startswith("rename:"):
            session_id = data.split(":", 1)[1]
            chat_data = self.sessions.setdefault(str(chat_id), {"sessions": []})
            chat_data["rename_pending"] = session_id
            save_sessions(self.sessions)
            self.api.send_message(chat_id, "✏️ Send me the new name for this session:")

        elif data.startswith("nav:"):
            new_page = int(data.split(":", 1)[1])
            chat_data = self.sessions.setdefault(str(chat_id), {})
            chat_data["sessions_page"] = new_page
            save_sessions(self.sessions)
            text, keyboard = self.build_session_message(chat_id, new_page)
            self.api.edit_message(chat_id, message_id, text, parse_mode="MarkdownV2")
            if keyboard:
                self.api._request(
                    "editMessageReplyMarkup",
                    chat_id=chat_id,
                    message_id=message_id,
                    reply_markup=json.dumps(keyboard),
                )

        elif data == "find_session":
            chat_data = self.sessions.setdefault(str(chat_id), {"sessions": []})
            chat_data["find_pending"] = True
            save_sessions(self.sessions)
            self.api.send_message(
                chat_id,
                "🔎 *Find Session*\n\n🪪 Send me the session ID you'd like to add:",
                parse_mode="MarkdownV2",
            )

        elif data.startswith("find_continue:"):
            session_id = data.split(":", 1)[1]
            self.set_active_session(chat_id, session_id)
            self.api.send_message(
                chat_id,
                "✅ *Session resumed*\n\nContinuing conversation\\.\\.\\.",
                parse_mode="MarkdownV2",
            )

        elif data == "find_cancel":
            self.handle_sessions(chat_id)

        elif data.startswith("stop_job:"):
            jid = int(data.split(":", 1)[1])
            with self._jobs_lock:
                job = self._jobs.get(jid)
            if job and job["thread"].is_alive():
                job["stop_event"].set()
                self.api.send_message(chat_id, "🛑 Cancelling request\\.\\.\\.")
            else:
                self.api.send_message(chat_id, "ℹ️ That request already finished.")

    def handle_stop(self, chat_id: int):
        with self._jobs_lock:
            active = {jid: j for jid, j in self._jobs.items() if j["chat_id"] == chat_id and j["thread"].is_alive()}
        if not active:
            self.api.send_message(chat_id, "✅ No active requests running.")
            return
        keyboard = {"inline_keyboard": [
            [{"text": f"🛑 {j['preview']}", "callback_data": f"stop_job:{jid}"}]
            for jid, j in active.items()
        ]}
        self.api.send_message(chat_id, "🛑 *Active requests* — tap to cancel:", parse_mode="MarkdownV2", reply_markup=keyboard)

    def _register_job(self, chat_id: int, preview: str, stop_event, thread, message_id=None) -> int:
        import threading
        with self._jobs_lock:
            self._job_counter += 1
            jid = self._job_counter
            self._jobs[jid] = {
                "thread": thread,
                "stop_event": stop_event,
                "chat_id": chat_id,
                "preview": preview[:40],
                "message_id": message_id,
            }
        return jid

    def _unregister_job(self, jid: int):
        with self._jobs_lock:
            self._jobs.pop(jid, None)

    def handle_message(self, message: dict):
        if not isinstance(message, dict):
            log(f"Invalid message format: {type(message)}")
            return

        chat_id = message.get("chat", {}).get("id") if isinstance(message.get("chat"), dict) else None
        text = message.get("text", "") if isinstance(message.get("text"), str) else ""

        if not chat_id:
            log("Message missing chat_id")
            return

        if not self.is_allowed(chat_id):
            log(f"Rejected message from unauthorized chat {chat_id}")
            return

        # Handle voice messages
        voice = message.get("voice")
        if voice and isinstance(voice, dict):
            caption = message.get("caption") if isinstance(message.get("caption"), str) else None
            self.process_voice_message(chat_id, voice, caption)
            return

        # Handle photo messages
        photos = message.get("photo")
        if photos and isinstance(photos, list):
            caption = message.get("caption") if isinstance(message.get("caption"), str) else None
            self.process_photo_message(chat_id, photos, caption)
            return

        if not text:
            return

        if text.startswith("/"):
            if text == "/start":
                self.handle_start(chat_id)
            elif text == "/new":
                self.handle_new(chat_id)
            elif text == "/sessions":
                self.handle_sessions(chat_id)
            elif text == "/skills":
                self.handle_skills(chat_id)
            elif text == "/stop":
                self.handle_stop(chat_id)
            return

        chat_data = self.sessions.setdefault(str(chat_id), {"sessions": []})

        # Handle find session pending - must check BEFORE other processing
        if chat_data.get("find_pending"):
            # Clear the flag immediately so we don't loop
            del chat_data["find_pending"]
            save_sessions(self.sessions)
            session_id = text.strip()
            self.handle_find_session_result(chat_id, session_id)
            return

        # Handle rename pending - must check BEFORE normal messages
        if chat_data.get("rename_pending"):
            rename_session_id = chat_data["rename_pending"]
            new_title = text.strip()
            # Clear the flag first
            del chat_data["rename_pending"]
            save_sessions(self.sessions)
            if self.rename_session(chat_id, rename_session_id, new_title):
                self.api.send_message(chat_id, f"✅ Renamed to: *{escape_telegram_markdown(new_title)}*", parse_mode="MarkdownV2")
                self.handle_sessions(chat_id)
            else:
                self.api.send_message(chat_id, "❌ Could not rename session")
            return

        log(f"Processing message from chat {chat_id}: {text[:50]}...")
        self.api.send_chat_action(chat_id, "typing")
        session_id = self.get_chat_session(chat_id)

        import threading as _threading
        def _run():
            try:
                response, returned_session_id = self.call_claude_with_streaming(chat_id, text, session_id)
                if returned_session_id and returned_session_id != session_id:
                    self.update_session_id(chat_id, session_id, returned_session_id, text)
                elif returned_session_id:
                    self.update_session_timestamp(chat_id, returned_session_id)
            except Exception as e:
                log(f"Error processing message: {e}")
                self.api.send_message(chat_id, f"❌ Error: {e}")
        _threading.Thread(target=_run, daemon=True).start()

    def _send_long_response(self, chat_id: int, text: str, edit_message_id: Optional[int] = None):
        """Send text, splitting into multiple messages if over Telegram limit."""
        chunks = truncate_for_telegram(text, max_length=4000)
        for idx, chunk in enumerate(chunks):
            display = escape_telegram_markdown(format_for_telegram(chunk))
            if idx == 0 and edit_message_id:
                result = self.api.edit_message(chat_id, edit_message_id, display, parse_mode="MarkdownV2")
                if not result.get("ok"):
                    self.api.send_message(chat_id, chunk[:4000])
            else:
                result = self.api.send_message(chat_id, display, parse_mode="MarkdownV2")
                if not result.get("ok"):
                    self.api.send_message(chat_id, chunk[:4000])

    def route_response(self, chat_id: int, response: str):
        """Route Claude's response based on format prefixes."""
        response = response.strip()

        # IMAGE: prefix - generate and send image
        if response.startswith("IMAGE:"):
            prompt = response[6:].strip()
            self.api.send_chat_action(chat_id, "upload_photo")
            img_path = self.media.generate_image(prompt)
            if img_path:
                self.api.send_photo(chat_id, img_path)
                self.media.cleanup(img_path)
            else:
                self.api.send_message(chat_id, "❌ Failed to generate image")
            return

        # VOICE: prefix - text to speech
        if response.startswith("VOICE:"):
            text = response[6:].strip()
            self.api.send_chat_action(chat_id, "record_voice")
            audio_path = self.media.text_to_speech(text)
            if audio_path:
                self.api.send_voice(chat_id, audio_path)
                self.media.cleanup(audio_path)
            else:
                self.api.send_message(chat_id, f"{text}\n\n_(TTS failed)_")
            return

        # Default: send as text
        display_text = format_for_telegram(response)
        display_text = escape_telegram_markdown(display_text)
        if len(display_text) > 4090:
            display_text = display_text[:4087] + "..."
        result = self.api.send_message(chat_id, display_text, parse_mode="MarkdownV2")
        if not result.get("ok"):
            self.api.send_message(chat_id, display_text[:4000])

    def process_voice_message(self, chat_id: int, voice: dict, caption: Optional[str] = None):
        """Process incoming voice message: transcribe and send to Claude."""
        file_id = voice.get("file_id")
        if not file_id:
            self.api.send_message(chat_id, "❌ Could not get voice file")
            return

        self.api.send_chat_action(chat_id, "typing")

        # Download voice file
        file_url = self.api.get_file(file_id)
        if not file_url:
            self.api.send_message(chat_id, "❌ Could not download voice")
            return

        audio_path = self.media.download_file(file_url)
        if not audio_path:
            self.api.send_message(chat_id, "❌ Download failed")
            return

        # Transcribe
        transcription = self.media.speech_to_text(audio_path)
        self.media.cleanup(audio_path)

        # Build context with transcription
        context = f"[User sent a voice message: \"{transcription}\"]"
        if caption:
            context += f"\n[Caption: {caption}]"

        # Send to Claude with system prompt hint for voice replies
        session_id = self.get_chat_session(chat_id)
        full_prompt = f"""The user sent you a voice message. Here's what they said:

\"{transcription}\"

{'' if not caption else f'Additional context: {caption}'}

You can respond with:
- Normal text (default)
- VOICE: your message here (if you want to reply with voice, good for casual/conversational replies)
- IMAGE: description here (if you want to generate an image)

Reply naturally:"""

        try:
            response, returned_session_id = self.call_claude(full_prompt, session_id)
            if returned_session_id and returned_session_id != session_id:
                self.update_session_id(chat_id, session_id, returned_session_id, transcription)
            elif returned_session_id:
                self.update_session_timestamp(chat_id, returned_session_id)
            self.route_response(chat_id, response)
        except Exception as e:
            log(f"Error processing voice: {e}")
            self.api.send_message(chat_id, f"❌ Error: {e}")

    def process_photo_message(self, chat_id: int, photos: list, caption: Optional[str] = None):
        """Process incoming photo: analyze and send to Claude."""
        # Get largest photo (last in list)
        photo = photos[-1] if photos else None
        if not photo:
            self.api.send_message(chat_id, "❌ Could not get photo")
            return

        file_id = photo.get("file_id")
        if not file_id:
            self.api.send_message(chat_id, "❌ Could not get photo file")
            return

        self.api.send_chat_action(chat_id, "typing")

        # Download photo
        file_url = self.api.get_file(file_id)
        if not file_url:
            self.api.send_message(chat_id, "❌ Could not download photo")
            return

        photo_path = self.media.download_file(file_url)
        if not photo_path:
            self.api.send_message(chat_id, "❌ Download failed")
            return

        # Analyze image
        analysis = self.media.analyze_image(photo_path)

        # Build context
        context = f"[User sent an image. Analysis: {analysis}]"
        if caption:
            context += f"\n[User's caption/question: {caption}]"

        # Send to Claude with system prompt hint
        session_id = self.get_chat_session(chat_id)
        full_prompt = f"""The user sent you an image. Here's what I see in it:

{analysis}

{'' if not caption else f'The user also said: \"{caption}\"'}

You can respond with:
- Normal text (describe, explain, answer questions)
- IMAGE: description here (if you want to generate a new image based on this one)
- VOICE: your message here (if you want to reply with voice)

Reply helpfully:"""

        try:
            response, returned_session_id = self.call_claude(full_prompt, session_id)
            if returned_session_id and returned_session_id != session_id:
                self.update_session_id(chat_id, session_id, returned_session_id, caption or "[image]")
            elif returned_session_id:
                self.update_session_timestamp(chat_id, returned_session_id)
            self.route_response(chat_id, response)
        except Exception as e:
            log(f"Error processing photo: {e}")
            self.api.send_message(chat_id, f"❌ Error: {e}")
        finally:
            self.media.cleanup(photo_path)

    def _is_duplicate(self, update_id: int) -> bool:
        """Check if update was already processed."""
        if update_id in self._processed_messages:
            return True
        self._processed_messages.add(update_id)
        # Keep set size bounded
        if len(self._processed_messages) > self._max_tracked:
            self._processed_messages = set(sorted(self._processed_messages)[-self._max_tracked//2:])
        return False

    def run(self):
        log("Starting Telegram Claude Bot...")
        log(f"Allowed chats: {ALLOWED_CHAT_IDS if ALLOWED_CHAT_IDS else 'all'}")

        while True:
            try:
                updates = self.api.get_updates(self.offset + 1)
                for update in updates:
                    if not isinstance(update, dict):
                        continue
                    update_id = update.get("update_id", 0)
                    if self._is_duplicate(update_id):
                        log(f"Skipping duplicate update {update_id}")
                        continue
                    self.offset = max(self.offset, update_id)
                    if "message" in update:
                        self.handle_message(update.get("message"))
                    elif "callback_query" in update:
                        self.handle_callback(update.get("callback_query"))
            except Exception as e:
                log(f"Error in main loop: {e}")
                time.sleep(5)


if __name__ == "__main__":
    bot = ClaudeBot()
    bot.run()
