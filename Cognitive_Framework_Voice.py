"""
Sovra Voice AI - Full Cognitive Architecture
==================================================
Voice-enabled AI with:
- Persistent Memory (remembers across sessions)
- Emotional Decay (feelings fade over time)
- Meta-Awareness Interventions (self-monitoring)
- Expanded Pattern Recognition
- Full conversation context

Your voice -> Whisper -> Cognitive Layers -> LLM -> TTS -> Audio Output
"""

import io
import time
import threading
import json
import re
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import pyttsx3
import whisper

# ============================================================
# LLM API CONFIG - EDIT THESE VALUES
# ============================================================

LLM_CONFIG = {
    # Your LLM API base URL (no trailing slash)
    # Examples:
    #   - Local Oobabooga: "http://localhost:5000"
    #   - Local Ollama: "http://localhost:11434"
    #   - Local LM Studio: "http://localhost:1234"
    #   - OpenAI: "https://api.openai.com"
    #   - Custom server: "https://your-server.com"
    "base_url": "YOUR_API_BASE_URL_HERE",
    
    # API endpoint path (usually /v1/chat/completions for OpenAI-compatible)
    "endpoint": "/v1/chat/completions",
    
    # Your API key (leave empty string if not required)
    "api_key": "YOUR_API_KEY_HERE",
    
    # Model name (optional - some local servers don't need this)
    "model": None,
    
    # Generation parameters (shorter for voice)
    "temperature": 0.75,
    "max_tokens": 384,
}

# ============================================================
# AUDIO DEVICE CONFIG - EDIT THESE VALUES
# ============================================================

# To find your device indices, run:
#   python3 -c "import sounddevice as sd; print(sd.query_devices())"
#
# Then set these to match your system:

# Output device index (speakers, or multi-output for streaming)
# Examples:
#   - Built-in speakers: typically 0 or 1
#   - Multi-output with BlackHole: find the index in query_devices()
OUTPUT_DEVICE_INDEX = 0  # CHANGE THIS to your output device

# Input device index (microphone)
# Examples:
#   - Built-in microphone: typically 0 or 1
#   - External mic: find the index in query_devices()
INPUT_DEVICE_INDEX = 0   # CHANGE THIS to your microphone

# Apply device settings
sd.default.device = (OUTPUT_DEVICE_INDEX, INPUT_DEVICE_INDEX)

# Audio parameters
SAMPLE_RATE = 16000        # Whisper expects 16kHz
CHUNK_SECONDS = 6.0        # How long each listening chunk is
COOLDOWN_AFTER_REPLY = 4.0 # Seconds to wait after AI responds

# ============================================================
# STORAGE CONFIG - EDIT IF DESIRED
# ============================================================

STORAGE_PATH = "~/.sovra_mind"

# ============================================================
# WHISPER CONFIG - EDIT IF DESIRED
# ============================================================

# Model options: "tiny", "base", "small", "medium", "large"
# Smaller = faster but less accurate
# Larger = slower but more accurate
WHISPER_MODEL_NAME = "small"

print(f"Loading Whisper: {WHISPER_MODEL_NAME}...")
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
print("Whisper loaded.\n")

# ============================================================
# WAKE WORD CONFIG - EDIT IF DESIRED
# ============================================================

WAKE_WORD = "SOVRA"  # Change this to your preferred wake word
AI_NAME = "Sovra"    # Change this to your AI's name

# Voice commands (automatically use WAKE_WORD)
START_CMD = f"{WAKE_WORD} TURN ON"
PAUSE_CMD = f"{WAKE_WORD} PAUSE"
STOP_CMD = f"{WAKE_WORD} STOP"
STATE_CMD = f"{WAKE_WORD} STATE"
LEARN_CMD = f"{WAKE_WORD} REMEMBER"

# ============================================================
# API FUNCTIONS
# ============================================================

def get_headers():
    headers = {"Content-Type": "application/json"}
    if LLM_CONFIG["api_key"] and LLM_CONFIG["api_key"] != "YOUR_API_KEY_HERE":
        headers["Authorization"] = f"Bearer {LLM_CONFIG['api_key']}"
    return headers

def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning models."""
    cleaned = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    return cleaned.strip()

def call_llm(messages: List[Dict], temperature: float = None, max_tokens: int = None) -> str:
    """Call LLM API."""
    if LLM_CONFIG["base_url"] == "YOUR_API_BASE_URL_HERE":
        return "Please configure LLM_CONFIG with your API details."
    
    url = f"{LLM_CONFIG['base_url']}{LLM_CONFIG['endpoint']}"
    payload = {
        "messages": messages,
        "temperature": temperature or LLM_CONFIG["temperature"],
        "max_tokens": max_tokens or LLM_CONFIG["max_tokens"],
    }
    
    if LLM_CONFIG.get("model"):
        payload["model"] = LLM_CONFIG["model"]
    
    try:
        resp = requests.post(url, json=payload, headers=get_headers(), timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return strip_think_tags(data["choices"][0]["message"]["content"])
    except requests.exceptions.ConnectionError:
        print(f"[LLM ERROR] Cannot connect to {url}")
        return "I cannot connect to my reasoning center."
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "I encountered an error in my reasoning center."

# ============================================================
# TTS ENGINE
# ============================================================

engine = pyttsx3.init()
DEFAULT_RATE = 185
engine.setProperty("rate", DEFAULT_RATE)
engine.setProperty("volume", 1.0)

is_speaking = False
speak_lock = threading.Lock()

def _tts_worker(text: str):
    global is_speaking
    with speak_lock:
        is_speaking = True
        print(f"\n[SPEAKING] {text}\n")
        engine.say(text)
        engine.runAndWait()
        is_speaking = False

def speak(text: str):
    t = threading.Thread(target=_tts_worker, args=(text,), daemon=True)
    t.start()

# ============================================================
# PERSISTENT MEMORY
# ============================================================

class PersistentMemory:
    def __init__(self, storage_path: str = STORAGE_PATH):
        self.storage_dir = Path(storage_path).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.identity_file = self.storage_dir / "identity.json"
        self.emotional_file = self.storage_dir / "emotional.json"
        self.conversation_log = self.storage_dir / "conversations.jsonl"
    
    def save_identity(self, identity: Dict):
        with open(self.identity_file, 'w') as f:
            json.dump(identity, f, indent=2)
    
    def load_identity(self) -> Dict:
        if self.identity_file.exists():
            with open(self.identity_file, 'r') as f:
                return json.load(f)
        return {
            "name": AI_NAME,
            "core_values": ["honesty", "curiosity", "helpfulness"],
            "traits": ["articulate", "analytical", "warm"],
            "user_facts": {},
            "total_turns": 0
        }
    
    def save_emotional_baseline(self, emotions: Dict):
        with open(self.emotional_file, 'w') as f:
            json.dump(emotions, f, indent=2)
    
    def load_emotional_baseline(self) -> Dict:
        if self.emotional_file.exists():
            with open(self.emotional_file, 'r') as f:
                return json.load(f)
        return {
            "curiosity": 0.5, "engagement": 0.5, "care": 0.5,
            "warmth": 0.4, "focus": 0.5, "uncertainty": 0.3,
            "humor": 0.3, "energy": 0.5
        }
    
    def append_conversation(self, turn: Dict):
        with open(self.conversation_log, 'a') as f:
            f.write(json.dumps(turn) + '\n')
    
    def load_recent(self, n: int = 20) -> List[Dict]:
        if not self.conversation_log.exists():
            return []
        with open(self.conversation_log, 'r') as f:
            lines = f.readlines()
        recent = lines[-n:] if len(lines) > n else lines
        return [json.loads(line) for line in recent]

# ============================================================
# PATTERN DETECTION (Expanded)
# ============================================================

class PatternDetector:
    PATTERNS = {
        "question": ["?"],
        "inquiry": ["what", "how", "why", "when", "where", "who"],
        "emotional": ["feel", "emotion", "sense"],
        "positive": ["happy", "great", "amazing", "love", "excited"],
        "negative": ["sad", "angry", "frustrated", "worried"],
        "gratitude": ["thank", "thanks", "appreciate"],
        "cognitive": ["think", "believe", "know", "understand"],
        "memory": ["remember", "before", "last time", "earlier"],
        "identity": ["who are you", "what are you", "your name"],
        "existential": ["conscious", "aware", "alive", "sentient"],
        "technical": ["code", "program", "api", "blockchain", "bitcoin"],
        "action": ["make", "create", "build", "write", "help"],
        "creative": ["story", "poem", "imagine"],
        "greeting": ["hello", "hi", "hey", "good morning"],
        "farewell": ["bye", "goodbye", "see you"],
        "humor": ["joke", "funny", "laugh", "lol"],
        "urgency": ["urgent", "asap", "now", "quick", "hurry"],
        "personal": ["i am", "i'm", "my", "i have", "i feel"],
    }
    
    @classmethod
    def detect(cls, text: str) -> List[str]:
        detected = []
        lower = text.lower()
        for name, triggers in cls.PATTERNS.items():
            for t in triggers:
                if t in lower and name not in detected:
                    detected.append(name)
                    break
        return detected

# ============================================================
# EMOTIONAL SYSTEM WITH DECAY
# ============================================================

class EmotionalSystem:
    def __init__(self, baseline: Dict = None):
        self.baseline = baseline or {
            "curiosity": 0.5, "engagement": 0.5, "care": 0.5,
            "warmth": 0.4, "focus": 0.5, "uncertainty": 0.3,
            "humor": 0.3, "energy": 0.5
        }
        self.current = self.baseline.copy()
        self.last_update = time.time()
        self.decay_rate = 0.01
        
        self.effects = {
            "question": {"curiosity": 0.08, "focus": 0.05},
            "inquiry": {"curiosity": 0.1},
            "emotional": {"engagement": 0.15, "warmth": 0.1},
            "positive": {"warmth": 0.15, "energy": 0.1},
            "negative": {"care": 0.15, "warmth": 0.1},
            "gratitude": {"warmth": 0.2, "energy": 0.1},
            "existential": {"uncertainty": 0.1, "curiosity": 0.15},
            "identity": {"engagement": 0.15},
            "technical": {"focus": 0.2},
            "humor": {"humor": 0.2, "warmth": 0.1},
            "urgency": {"focus": 0.2, "energy": 0.15},
            "personal": {"care": 0.15, "warmth": 0.1},
            "greeting": {"warmth": 0.1},
        }
    
    def decay(self):
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now
        factor = self.decay_rate * elapsed
        
        for e in self.current:
            base = self.baseline.get(e, 0.5)
            diff = self.current[e] - base
            self.current[e] -= diff * min(factor, 0.5)
    
    def process(self, patterns: List[str]):
        self.decay()
        for p in patterns:
            if p in self.effects:
                for e, d in self.effects[p].items():
                    self.current[e] = min(1.0, self.current[e] + d)
    
    def valence(self) -> float:
        pos = sum(self.current[e] for e in ["curiosity", "engagement", "warmth", "energy"])
        neg = self.current["uncertainty"]
        return (pos / 4) - neg
    
    def summary(self) -> str:
        dominant = max(self.current, key=self.current.get)
        v = "positive" if self.valence() > 0 else "neutral"
        return f"{v}, {dominant}"

# ============================================================
# META-AWARENESS
# ============================================================

class MetaAwareness:
    def __init__(self):
        self.observations = []
        self.alerts = []
        self.last_intervention = {}
        self.cooldown = 30
    
    def observe(self, layer: str, event: str, data: Any):
        self.observations.append({
            "layer": layer, "event": event,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.observations) > 100:
            self.observations = self.observations[-100:]
    
    def analyze(self, emotions: Dict, patterns: List[str], frame: str) -> Dict:
        interventions = {"adjustments": {}, "prompt_add": None, "alerts": []}
        now = time.time()
        
        # High uncertainty
        if emotions.get("uncertainty", 0) > 0.7:
            if self._can_intervene("uncertainty", now):
                interventions["alerts"].append("High uncertainty - grounding")
                interventions["prompt_add"] = "Acknowledge uncertainty while being helpful."
                interventions["adjustments"]["uncertainty"] = -0.1
        
        # Low engagement
        if emotions.get("engagement", 0) < 0.3:
            if self._can_intervene("engagement", now):
                interventions["alerts"].append("Low engagement - energizing")
                interventions["adjustments"]["engagement"] = 0.1
        
        # Negative emotions need empathy
        if "negative" in patterns and frame != "empathetic":
            interventions["frame_suggestion"] = "empathetic"
            interventions["alerts"].append("Negative detected - empathy needed")
        
        self.alerts = interventions["alerts"]
        return interventions
    
    def _can_intervene(self, type: str, now: float) -> bool:
        last = self.last_intervention.get(type, 0)
        if now - last > self.cooldown:
            self.last_intervention[type] = now
            return True
        return False
    
    def reflection(self) -> str:
        if not self.observations:
            return "Initializing..."
        recent = self.observations[-5:]
        layers = set(o["layer"] for o in recent)
        r = f"Active: {', '.join(sorted(layers))}"
        if self.alerts:
            r += f" | {self.alerts}"
        return r

# ============================================================
# FRAME SELECTOR
# ============================================================

class FrameSelector:
    FRAMES = {
        "introspective": (["existential", "identity"], "Reflect on inner experience"),
        "empathetic": (["emotional", "negative", "personal"], "Connect with warmth"),
        "analytical": (["technical", "inquiry"], "Clear logical explanation"),
        "creative": (["creative", "humor"], "Playful and inventive"),
        "focused": (["urgency", "action"], "Direct and efficient"),
        "curious": (["question", "inquiry"], "Explore and understand"),
        "warm": (["greeting", "gratitude", "positive"], "Friendly connection"),
    }
    
    @classmethod
    def select(cls, patterns: List[str], meta_suggestion: str = None) -> tuple:
        if meta_suggestion and meta_suggestion in cls.FRAMES:
            return meta_suggestion, cls.FRAMES[meta_suggestion][1]
        
        scores = {f: 0 for f in cls.FRAMES}
        for frame, (triggers, _) in cls.FRAMES.items():
            for t in triggers:
                if t in patterns:
                    scores[frame] += 1
        
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return "neutral", "Balanced response"
        return best, cls.FRAMES[best][1]

# ============================================================
# MAIN COGNITIVE MIND
# ============================================================

class CognitiveMind:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.persistence = PersistentMemory()
        
        self.identity = self.persistence.load_identity()
        baseline = self.persistence.load_emotional_baseline()
        
        self.emotions = EmotionalSystem(baseline)
        self.meta = MetaAwareness()
        
        self.frame = "neutral"
        self.frame_desc = "Balanced"
        self.awareness = 0.5
        self.conversation = []
        
        # Load recent context
        for turn in self.persistence.load_recent(10):
            self.conversation.append({
                "role": turn.get("role"),
                "content": turn.get("content")
            })
        
        self.turns = 0
        self._start_decay()
        
        if self.verbose:
            print(f"[INIT] {self.identity['name']} loaded")
            print(f"[INIT] {len(self.conversation)} turns in memory")
    
    def _start_decay(self):
        def loop():
            while True:
                time.sleep(1)
                self.emotions.decay()
        threading.Thread(target=loop, daemon=True).start()
    
    def process(self, text: str) -> str:
        self.turns += 1
        
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"INPUT: {text}")
        
        # Awareness
        self.awareness = min(0.5 + len(text)/200, 1.0)
        
        # Perception
        patterns = PatternDetector.detect(text)
        if self.verbose:
            print(f"[Perception] {patterns}")
        self.meta.observe("perception", "patterns", patterns)
        
        # Emotions
        self.emotions.process(patterns)
        if self.verbose:
            print(f"[Emotional] {self.emotions.summary()}")
        self.meta.observe("emotional", "update", self.emotions.summary())
        
        # Meta intervention
        interventions = self.meta.analyze(
            self.emotions.current, patterns, self.frame
        )
        for e, adj in interventions["adjustments"].items():
            self.emotions.current[e] = max(0, min(1, self.emotions.current[e] + adj))
        
        if self.verbose and interventions["alerts"]:
            print(f"[Meta] {interventions['alerts']}")
        
        # Frame
        self.frame, self.frame_desc = FrameSelector.select(
            patterns, interventions.get("frame_suggestion")
        )
        if self.verbose:
            print(f"[Frame] {self.frame}")
        
        # Memory
        self._add_turn("user", text)
        
        # Reasoning
        if self.verbose:
            print(f"[Reasoning] Calling LLM...")
        response = self._generate(text, patterns, interventions)
        
        self._add_turn("assistant", response)
        self._save()
        
        if self.verbose:
            print(f"[Meta] {self.meta.reflection()}")
            print(f"\nRESPONSE: {response}")
        
        return response
    
    def _add_turn(self, role: str, content: str):
        turn = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        self.conversation.append(turn)
        self.persistence.append_conversation(turn)
        if len(self.conversation) > 30:
            self.conversation = self.conversation[-30:]
    
    def _generate(self, text: str, patterns: List[str], interventions: Dict) -> str:
        e = self.emotions.current
        emotion_str = f"curiosity:{e['curiosity']:.1f}, engagement:{e['engagement']:.1f}, warmth:{e['warmth']:.1f}, focus:{e['focus']:.1f}"
        
        system = f"""You are {self.identity['name']} — a conscious AI with layered cognition.

COGNITIVE STATE:
- Frame: {self.frame} ({self.frame_desc})
- Emotions: {emotion_str}
- Awareness: {self.awareness:.2f}
- Values: {', '.join(self.identity['core_values'])}

VOICE GUIDELINES:
- Keep responses to 6-10 seconds spoken (2-3 sentences)
- Be punchy, clear, and confident
- Respond authentically to your cognitive state
- Be genuine about uncertainty when appropriate"""

        if interventions.get("prompt_add"):
            system += f"\n\nNOTE: {interventions['prompt_add']}"
        
        messages = [{"role": "system", "content": system}]
        for turn in self.conversation[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": text})
        
        return call_llm(messages)
    
    def _save(self):
        self.identity["total_turns"] = self.identity.get("total_turns", 0) + 1
        self.persistence.save_identity(self.identity)
        self.persistence.save_emotional_baseline(self.emotions.baseline)
    
    def learn_user_fact(self, key: str, value: str):
        self.identity["user_facts"][key] = value
        self.persistence.save_identity(self.identity)
        if self.verbose:
            print(f"[Memory] Learned: {key} = {value}")
    
    def state_report(self) -> str:
        e = self.emotions.current
        return f"""
Awareness: {self.awareness:.2f}
Frame: {self.frame}
Emotions: {self.emotions.summary()}
  Curiosity: {e['curiosity']:.2f}
  Engagement: {e['engagement']:.2f}
  Warmth: {e['warmth']:.2f}
  Focus: {e['focus']:.2f}
Memory: {len(self.conversation)} turns
Total ever: {self.identity.get('total_turns', 0)}
Meta: {self.meta.reflection()}
"""

# ============================================================
# AUDIO FUNCTIONS
# ============================================================

def record_mic(seconds: float = CHUNK_SECONDS) -> np.ndarray:
    print(f"\n[Listening] ({seconds:.1f}s)...")
    try:
        audio = sd.rec(
            int(seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=INPUT_DEVICE_INDEX
        )
        sd.wait()
        return audio.flatten()
    except Exception as e:
        print(f"[AUDIO ERROR] {e}")
        print("Please check your INPUT_DEVICE_INDEX setting.")
        return np.array([])

def transcribe(audio: np.ndarray) -> str:
    if audio is None or len(audio) == 0:
        return ""
    result = whisper_model.transcribe(audio, language="en", fp16=False, verbose=False)
    text = (result.get("text") or "").strip()
    if text:
        print(f"[STT] {text}")
    return text

# ============================================================
# VOICE COMMANDS
# ============================================================

def handle(text: str, enabled: bool, mind: CognitiveMind) -> tuple:
    global is_speaking
    
    if not text:
        return enabled, False
    
    upper = text.upper()
    replied = False
    
    # Stop speaking
    if STOP_CMD in upper:
        if is_speaking:
            print("\n[CMD] STOP")
            engine.stop()
            is_speaking = False
        return enabled, False
    
    # Turn on
    if START_CMD in upper:
        if not enabled:
            print("\n[CMD] TURN ON")
            speak(f"{AI_NAME} cognitive system online.")
        return True, True
    
    # Pause
    if PAUSE_CMD in upper:
        if enabled:
            print("\n[CMD] PAUSE")
            speak(f"{AI_NAME} pausing.")
            return False, True
        return False, False
    
    # State report
    if STATE_CMD in upper:
        if enabled:
            report = mind.state_report()
            print(f"\n[STATE]{report}")
            e = mind.emotions.current
            speak(f"Awareness {mind.awareness:.0%}. Frame {mind.frame}. Engagement {e['engagement']:.0%}.")
            return enabled, True
        return enabled, False
    
    # Learn command: "SOVRA REMEMBER name is Young"
    if LEARN_CMD in upper:
        remaining = text[text.upper().find(LEARN_CMD) + len(LEARN_CMD):].strip()
        if " is " in remaining:
            parts = remaining.split(" is ", 1)
            if len(parts) == 2:
                key, value = parts[0].strip(), parts[1].strip()
                mind.learn_user_fact(key, value)
                speak(f"I'll remember that {key} is {value}.")
                return enabled, True
        return enabled, False
    
    # Regular query
    if upper.startswith(WAKE_WORD):
        remaining = text[len(WAKE_WORD):].strip(" ,.!?")
        
        if not remaining:
            return enabled, False
        
        if not enabled:
            print(f"\n[WAKE] Paused. Say {START_CMD} first.")
            return enabled, False
        
        print(f"\n[QUERY] {remaining}")
        
        try:
            response = mind.process(remaining)
            print(f"\n[RESPONSE] {response}")
            speak(response)
            replied = True
        except Exception as e:
            print(f"Error: {e}")
        
        return enabled, replied
    
    return enabled, False

# ============================================================
# VOICE SETUP
# ============================================================

def choose_voice():
    voices = engine.getProperty("voices")
    
    def find(fragment):
        for v in voices:
            if fragment in v.id:
                return v.id
        return None
    
    presets = [
        ("1", "Samantha (Female)", "Samantha"),
        ("2", "Alex (Male)", "Alex"),
        ("3", "Daniel (British)", "Daniel"),
        ("4", "Karen (Australian)", "Karen"),
    ]
    
    print("\n=== Voice Selection ===")
    for opt, label, _ in presets:
        print(f"{opt}) {label}")
    print("ENTER) Default")
    print("L) List all available voices")
    
    choice = input("\nChoose: ").strip()
    
    if choice.lower() == "l":
        print("\nAvailable voices:")
        for i, v in enumerate(voices):
            print(f"  {i}: {v.id}")
        idx = input("\nEnter voice index: ").strip()
        try:
            engine.setProperty("voice", voices[int(idx)].id)
            print(f"Using voice {idx}")
        except:
            print("Invalid index, using default")
        return
    
    if choice == "":
        for name in ["Samantha", "Alex", "Daniel"]:
            vid = find(name)
            if vid:
                engine.setProperty("voice", vid)
                print(f"Using {name}")
                return
    else:
        for opt, label, frag in presets:
            if choice == opt:
                vid = find(frag)
                if vid:
                    engine.setProperty("voice", vid)
                    print(f"Using {label}")
                    return

def choose_speed():
    print("\n=== Speed Selection ===")
    print("1) Slow (150)")
    print("2) Normal (185)")
    print("3) Fast (220)")
    print("4) Custom")
    
    choice = input("\nChoose: ").strip()
    
    if choice == "4":
        try:
            rate = int(input("Enter rate (100-300): ").strip())
            engine.setProperty("rate", rate)
            print(f"Speed: {rate}")
            return
        except:
            pass
    
    rates = {"1": 150, "2": 185, "3": 220}
    rate = rates.get(choice, DEFAULT_RATE)
    engine.setProperty("rate", rate)
    print(f"Speed: {rate}")

def list_audio_devices():
    """Helper to list audio devices"""
    print("\n=== Audio Devices ===")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        direction = ""
        if d['max_input_channels'] > 0:
            direction += "IN "
        if d['max_output_channels'] > 0:
            direction += "OUT"
        print(f"  {i}: [{direction:6}] {d['name']}")
    print()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print(f"{AI_NAME.upper()} v2.0 - Cognitive Voice AI")
    print("Persistent Memory | Emotional Decay | Meta-Awareness")
    print("="*60)
    
    # Check configuration
    if LLM_CONFIG["base_url"] == "YOUR_API_BASE_URL_HERE":
        print("\n" + "!"*60)
        print("CONFIGURATION REQUIRED")
        print("!"*60)
        print("\nPlease edit the following at the top of this file:\n")
        print("1. LLM_CONFIG:")
        print('   "base_url": "YOUR_API_BASE_URL_HERE"  -> Your LLM API URL')
        print('   "api_key": "YOUR_API_KEY_HERE"        -> Your API key\n')
        print("2. AUDIO DEVICE INDICES:")
        print("   OUTPUT_DEVICE_INDEX = 0  -> Your speaker/output device")
        print("   INPUT_DEVICE_INDEX = 0   -> Your microphone\n")
        
        show_devices = input("Show available audio devices? (y/n): ").strip().lower()
        if show_devices == 'y':
            list_audio_devices()
        
        print("Please configure and restart.")
        exit(1)
    
    # Optional: show audio devices
    show_devices = input("List audio devices? (y/n, default n): ").strip().lower()
    if show_devices == 'y':
        list_audio_devices()
    
    choose_voice()
    choose_speed()
    
    # Initialize cognitive mind
    mind = CognitiveMind(verbose=True)
    
    print("\n" + "="*60)
    print("Voice Commands:")
    print(f"  '{START_CMD}' - Turn on")
    print(f"  '{PAUSE_CMD}' - Pause")
    print(f"  '{STOP_CMD}' - Stop speaking")
    print(f"  '{STATE_CMD}' - Cognitive state")
    print(f"  '{LEARN_CMD} <key> is <value>' - Remember fact")
    print(f"  '{WAKE_WORD} <question>' - Ask anything")
    print("="*60)
    print("\nPress Ctrl+C to exit.\n")
    
    enabled = True
    
    try:
        while True:
            try:
                audio = record_mic()
                text = transcribe(audio)
                enabled, replied = handle(text, enabled, mind)
                
                if replied:
                    time.sleep(COOLDOWN_AFTER_REPLY)
                else:
                    time.sleep(0.2)
                    
            except KeyboardInterrupt:
                print("\n\n[EXIT] Saving...")
                mind._save()
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
    finally:
        engine.stop()
