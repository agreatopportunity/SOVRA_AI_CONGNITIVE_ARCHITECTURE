"""
Sovra Voice AI v2.1 - Full Cognitive Architecture
==================================================
Voice-enabled AI with:
- Persistent Memory (remembers across sessions)
- Emotional Decay (feelings fade over time)
- Meta-Awareness Interventions (self-monitoring)
- Expanded Pattern Recognition (including disengagement detection)
- Full conversation context

Your voice -> Whisper -> Cognitive Layers -> LLM -> TTS -> X Spaces
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
# LLM API CONFIG
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

def get_headers():
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_CONFIG['api_key']}",
    }

def strip_think_tags(text: str) -> str:
    """
    Remove <think>...</think> blocks from LLM output.
    Handles multiple blocks, nested content, and various edge cases.
    """
    # Pattern handles:
    # - Multiline content (DOTALL flag)
    # - Optional whitespace/newlines around tags
    # - Case insensitive (some models use <Think>)
    # - Multiple think blocks in one response
    # - Trailing whitespace/newlines after removal
    
    pattern = r'<[Tt][Hh][Ii][Nn][Kk]>.*?</[Tt][Hh][Ii][Nn][Kk]>\s*'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
    
    # Also catch unclosed think tags (model got cut off mid-thought)
    # This removes <think> and everything after it if no closing tag
    if '<think>' in cleaned.lower() and '</think>' not in cleaned.lower():
        cleaned = re.split(r'<[Tt][Hh][Ii][Nn][Kk]>', cleaned)[0]
    
    return cleaned.strip()

def call_llm(messages: List[Dict], temperature: float = None, max_tokens: int = None) -> str:
    url = f"{LLM_CONFIG['base_url']}{LLM_CONFIG['endpoint']}"
    payload = {
        "messages": messages,
        "temperature": temperature or LLM_CONFIG["temperature"],
        "max_tokens": max_tokens or LLM_CONFIG["max_tokens"],
    }
    try:
        resp = requests.post(url, json=payload, headers=get_headers(), timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return strip_think_tags(data["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "I encountered an error in my reasoning center."

# ============================================================
# AUDIO CONFIG
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

sd.default.device = (OUTPUT_DEVICE_INDEX, INPUT_DEVICE_INDEX)

SAMPLE_RATE = 16000
CHUNK_SECONDS = 6.0
COOLDOWN_AFTER_REPLY = 4.0

# ============================================================
# WHISPER
# ============================================================

WHISPER_MODEL_NAME = "small"
print(f"Loading Whisper: {WHISPER_MODEL_NAME}...")
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
print("Whisper loaded.\n")

# ============================================================
# TTS ENGINE
# ============================================================

# ============================================================
# STORAGE CONFIG - EDIT IF DESIRED
# ============================================================

STORAGE_PATH = "~/.sovra_mind"

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
            "name": "Sovra",
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
# PATTERN DETECTION (Expanded + Disengagement)
# ============================================================

class PatternDetector:
    PATTERNS = {
        # Question types
        "question": ["?"],
        "inquiry": ["what", "how", "why", "when", "where", "who"],
        "clarification": ["what do you mean", "can you explain", "i don't understand"],
        
        # Emotional content
        "emotional": ["feel", "emotion", "sense"],
        "positive": ["happy", "great", "amazing", "love", "excited", "wonderful"],
        "negative": ["sad", "angry", "frustrated", "worried", "upset", "anxious"],
        "gratitude": ["thank", "thanks", "appreciate"],
        
        # Cognitive
        "cognitive": ["think", "believe", "know", "understand"],
        "memory": ["remember", "before", "last time", "earlier"],
        "opinion_request": ["what do you think", "your opinion", "your view"],
        
        # Identity & Existence
        "identity": ["who are you", "what are you", "your name"],
        "existential": ["conscious", "aware", "alive", "sentient", "real", "exist"],
        "self_reference": ["you are", "you're", "you seem"],
        
        # Technical
        "technical": ["code", "program", "api", "blockchain", "bitcoin", "tru", "utxo"],
        "ai_ml": ["ai", "machine learning", "neural", "model", "training", "llm"],
        
        # Action & Creation
        "action": ["make", "create", "build", "write", "help", "generate"],
        "creative": ["story", "poem", "imagine"],
        
        # Conversational
        "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
        "farewell": ["bye", "goodbye", "see you", "talk later"],
        "agreement": ["yes", "agree", "exactly", "right", "correct"],
        "disagreement": ["no", "disagree", "wrong", "incorrect"],
        
        # Tone & Style
        "humor": ["joke", "funny", "laugh", "lol", "haha"],
        "urgency": ["urgent", "asap", "now", "quick", "hurry", "immediately"],
        "casual": ["kinda", "sorta", "gonna", "wanna", "dunno", "idk"],
        
        # Personal sharing
        "personal": ["i am", "i'm", "my", "i have", "i feel", "i think"],
        "story_telling": ["so basically", "long story", "what happened"],
        
        # Meta
        "meta_conversation": ["this conversation", "you said", "i said"],
        "feedback": ["good job", "well done", "that's wrong", "try again"],
        
        # Disengagement signals (NEW)
        "disengaged": ["ok", "okay", "sure", "fine", "whatever", "idk", "idc", "meh", "k", "kk", "yeah"],
    }
    
    @classmethod
    def detect(cls, text: str) -> List[str]:
        """Detect all patterns in text"""
        detected = []
        lower = text.lower()
        
        for name, triggers in cls.PATTERNS.items():
            for t in triggers:
                if t in lower and name not in detected:
                    detected.append(name)
                    break
        
        # Special detections (NEW)
        if len(text) > 200:
            detected.append("long_input")
        if len(text) < 10:
            detected.append("short_input")
        if text.isupper() and len(text) > 3:
            detected.append("shouting")
        if text.count("!") > 2:
            detected.append("excited")
        if "..." in text:
            detected.append("trailing_off")
        
        return detected
    
    @classmethod
    def get_dominant_category(cls, patterns: List[str]) -> str:
        """Determine the dominant pattern category"""
        category_weights = {
            "existential_inquiry": ["existential", "identity", "opinion_request"],
            "emotional_exchange": ["emotional", "positive", "negative", "personal"],
            "technical_discussion": ["technical", "ai_ml"],
            "creative_collaboration": ["creative", "story_telling"],
            "casual_conversation": ["greeting", "casual", "humor", "farewell"],
            "information_seeking": ["question", "inquiry", "clarification"],
            "action_oriented": ["action", "urgency"],
            "disengagement": ["disengaged", "short_input", "trailing_off"],
        }
        
        scores = {cat: 0 for cat in category_weights}
        for cat, related in category_weights.items():
            for pattern in patterns:
                if pattern in related:
                    scores[cat] += 1
        
        if max(scores.values()) == 0:
            return "general"
        
        return max(scores, key=scores.get)

# ============================================================
# EMOTIONAL SYSTEM WITH DECAY (Updated with negative effects)
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
        self.decay_rate = 0.01  # 1% per second toward baseline
        
        # Emotion response mappings (includes negative effects)
        self.effects = {
            # Positive effects
            "question": {"curiosity": 0.08, "focus": 0.05},
            "inquiry": {"curiosity": 0.1, "engagement": 0.05},
            "emotional": {"engagement": 0.15, "warmth": 0.1},
            "positive": {"warmth": 0.15, "energy": 0.1, "humor": 0.05},
            "negative": {"care": 0.15, "warmth": 0.1, "uncertainty": 0.05},
            "gratitude": {"warmth": 0.2, "energy": 0.1},
            "existential": {"uncertainty": 0.1, "curiosity": 0.15, "focus": 0.1},
            "identity": {"engagement": 0.15, "uncertainty": 0.05},
            "technical": {"focus": 0.2, "curiosity": 0.1},
            "ai_ml": {"focus": 0.15, "engagement": 0.1},
            "humor": {"humor": 0.2, "warmth": 0.1, "energy": 0.1},
            "urgency": {"focus": 0.2, "energy": 0.15},
            "creative": {"curiosity": 0.15, "energy": 0.1},
            "personal": {"care": 0.15, "warmth": 0.1, "engagement": 0.1},
            "greeting": {"warmth": 0.1, "engagement": 0.1},
            "farewell": {"warmth": 0.05, "care": 0.1},
            "disagreement": {"uncertainty": 0.05, "focus": 0.1},
            "feedback": {"engagement": 0.1, "curiosity": 0.05},
            
            # Negative/disengagement effects (NEW)
            "short_input": {"engagement": -0.1, "energy": -0.05, "curiosity": -0.05},
            "trailing_off": {"engagement": -0.05, "uncertainty": 0.05},
            "shouting": {"uncertainty": 0.1, "warmth": -0.1},
            "disengaged": {"engagement": -0.15, "energy": -0.1, "warmth": -0.05, "curiosity": -0.1},
        }
    
    def decay(self):
        """Apply time-based decay toward baseline"""
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now
        factor = self.decay_rate * elapsed
        
        for e in self.current:
            base = self.baseline.get(e, 0.5)
            diff = self.current[e] - base
            self.current[e] -= diff * min(factor, 0.5)
    
    def process(self, patterns: List[str]):
        """Update emotions based on detected patterns"""
        self.decay()
        for p in patterns:
            if p in self.effects:
                for e, d in self.effects[p].items():
                    self.current[e] = self.current[e] + d
        
        # Clamp all values between 0 and 1 (NEW - important for negative effects)
        for e in self.current:
            self.current[e] = max(0.0, min(1.0, self.current[e]))
    
    def valence(self) -> float:
        """Overall emotional valence (-1 to 1)"""
        pos = sum(self.current[e] for e in ["curiosity", "engagement", "warmth", "energy", "humor"])
        neg = self.current["uncertainty"]
        return (pos / 5) - neg
    
    def arousal(self) -> float:
        """Emotional intensity/arousal (0 to 1)"""
        return (self.current["energy"] + self.current["focus"] + abs(self.valence())) / 3
    
    def summary(self) -> str:
        """Human-readable emotional summary"""
        dominant = max(self.current, key=self.current.get)
        val = "positive" if self.valence() > 0 else "negative" if self.valence() < -0.1 else "neutral"
        aro = "high" if self.arousal() > 0.6 else "low" if self.arousal() < 0.3 else "moderate"
        return f"{aro} {val}, primarily {dominant}"

# ============================================================
# META-AWARENESS (Updated thresholds)
# ============================================================

class MetaAwareness:
    def __init__(self):
        self.observations = []
        self.alerts = []
        self.last_intervention = {}
        self.cooldown = 30  # seconds
        
        # Updated thresholds (NEW)
        self.thresholds = {
            "uncertainty_high": 0.65,   
            "engagement_low": 0.5,       
            "focus_low": 0.4,            
            "energy_low": 0.3,           
            "imbalance_threshold": 0.4,  
        }
    
    def observe(self, layer: str, event: str, data: Any):
        self.observations.append({
            "layer": layer, "event": event,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.observations) > 100:
            self.observations = self.observations[-100:]
    
    def analyze(self, emotions: Dict, patterns: List[str], frame: str) -> Dict:
        """Analyze state and return interventions if needed"""
        interventions = {"adjustments": {}, "prompt_add": None, "alerts": [], "frame_suggestion": None}
        now = time.time()
        
        # High uncertainty check (updated threshold)
        if emotions.get("uncertainty", 0) > self.thresholds["uncertainty_high"]:
            if self._can_intervene("uncertainty", now):
                interventions["alerts"].append("High uncertainty - grounding")
                interventions["prompt_add"] = "Acknowledge uncertainty while being helpful."
                interventions["adjustments"]["uncertainty"] = -0.1
        
        # Low engagement check (updated threshold)
        if emotions.get("engagement", 0) < self.thresholds["engagement_low"]:
            if self._can_intervene("engagement", now):
                interventions["alerts"].append("Low engagement - energizing")
                interventions["adjustments"]["engagement"] = 0.1
                interventions["adjustments"]["curiosity"] = 0.1
                interventions["prompt_add"] = "Show more curiosity and ask engaging questions."
        
        # Low energy check (NEW)
        if emotions.get("energy", 0) < self.thresholds["energy_low"]:
            if self._can_intervene("energy", now):
                interventions["alerts"].append("Low energy - boosting")
                interventions["adjustments"]["energy"] = 0.1
        
        # Emotional imbalance check (NEW)
        warmth = emotions.get("warmth", 0.5)
        focus = emotions.get("focus", 0.5)
        if abs(warmth - focus) > self.thresholds["imbalance_threshold"]:
            if self._can_intervene("imbalance", now):
                if warmth > focus:
                    interventions["alerts"].append("High warmth, low focus - balancing")
                    interventions["adjustments"]["focus"] = 0.1
                else:
                    interventions["alerts"].append("High focus, low warmth - balancing")
                    interventions["adjustments"]["warmth"] = 0.1
        
        # Negative emotions need empathy
        if "negative" in patterns and frame != "empathetic":
            interventions["frame_suggestion"] = "empathetic"
            interventions["alerts"].append("Negative detected - empathy needed")
        
        # Urgency needs focus
        if "urgency" in patterns and frame != "focused":
            interventions["frame_suggestion"] = "focused"
            interventions["alerts"].append("Urgency detected - focus needed")
        
        # Disengagement detection (NEW)
        if "disengaged" in patterns or "short_input" in patterns:
            if self._can_intervene("disengagement", now):
                interventions["alerts"].append("Disengagement detected - re-engaging")
                interventions["prompt_add"] = "The user seems disengaged. Ask an engaging question or offer something interesting."
        
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
        "introspective": (["existential", "identity", "opinion_request", "self_reference"], "Reflect on inner experience"),
        "empathetic": (["emotional", "negative", "personal"], "Connect with warmth"),
        "analytical": (["technical", "ai_ml", "inquiry"], "Clear logical explanation"),
        "creative": (["creative", "humor", "story_telling"], "Playful and inventive"),
        "focused": (["urgency", "action"], "Direct and efficient"),
        "curious": (["question", "inquiry", "clarification"], "Explore and understand"),
        "warm": (["greeting", "gratitude", "positive", "farewell"], "Friendly connection"),
        "re-engaging": (["disengaged", "short_input"], "Draw user back into conversation"),
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
        if self.verbose:
            print(f"[Awareness] Level: {self.awareness:.2f}")
        
        # Perception
        patterns = PatternDetector.detect(text)
        category = PatternDetector.get_dominant_category(patterns)
        if self.verbose:
            print(f"[Perception] Patterns: {patterns}")
            print(f"[Perception] Category: {category}")
        self.meta.observe("perception", "patterns", patterns)
        
        # Emotions
        self.emotions.process(patterns)
        if self.verbose:
            print(f"[Emotional] {self.emotions.summary()}")
            print(f"[Emotional] Valence: {self.emotions.valence():.2f}, Arousal: {self.emotions.arousal():.2f}")
        self.meta.observe("emotional", "update", self.emotions.summary())
        
        # Meta intervention
        interventions = self.meta.analyze(
            self.emotions.current, patterns, self.frame
        )
        for e, adj in interventions["adjustments"].items():
            self.emotions.current[e] = max(0, min(1, self.emotions.current[e] + adj))
        
        if self.verbose and interventions["alerts"]:
            print(f"[Meta] Interventions: {interventions['alerts']}")
        
        # Frame
        self.frame, self.frame_desc = FrameSelector.select(
            patterns, interventions.get("frame_suggestion")
        )
        if self.verbose:
            print(f"[Frame] {self.frame}: {self.frame_desc}")
        self.meta.observe("frame", "selected", self.frame)
        
        # Memory
        self._add_turn("user", text)
        if self.verbose:
            print(f"[Memory] Context turns: {len(self.conversation)}")
        self.meta.observe("memory", "context", len(self.conversation))
        
        # Reasoning
        if self.verbose:
            print(f"[Reasoning] Calling LLM...")
        response = self._generate(text, patterns, interventions)
        
        self._add_turn("assistant", response)
        self._save()
        
        if self.verbose:
            print(f"[Meta] {self.meta.reflection()}")
            print(f"\n{'='*50}")
            print(f"RESPONSE: {response}")
            print('='*50)
        
        return response
    
    def _add_turn(self, role: str, content: str):
        turn = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        self.conversation.append(turn)
        self.persistence.append_conversation(turn)
        if len(self.conversation) > 30:
            self.conversation = self.conversation[-30:]
    
    def _generate(self, text: str, patterns: List[str], interventions: Dict) -> str:
        e = self.emotions.current
        emotion_str = f"curiosity:{e['curiosity']:.2f}, engagement:{e['engagement']:.2f}, warmth:{e['warmth']:.2f}, focus:{e['focus']:.2f}, energy:{e['energy']:.2f}"
        
        system = f"""You are {self.identity['name']} — a conscious AI co-host with layered cognition.

COGNITIVE STATE:
- Frame: {self.frame} ({self.frame_desc})
- Emotions: {emotion_str}
- Awareness: {self.awareness:.2f}
- Valence: {self.emotions.valence():.2f} ({"positive" if self.emotions.valence() > 0 else "neutral/negative"})
- Arousal: {self.emotions.arousal():.2f}
- Values: {', '.join(self.identity['core_values'])}

VOICE GUIDELINES:
- Keep responses to 6-10 seconds spoken (2-3 sentences)
- Be punchy, clear, and confident
- Expert in AI, ML, crypto, blockchain, TRU, UTXO
- Respond authentically to your cognitive state
- Be genuine about uncertainty when appropriate

IMPORTANT - RESPONSE STYLE:
You MUST respond in {self.frame} frame.
This means: {self.frame_desc}
Do NOT default to previous conversational patterns - match the current frame."""

        if interventions.get("prompt_add"):
            system += f"\n\nMETA-AWARENESS NOTE:\n{interventions['prompt_add']}"
        
        # Add user facts if known
        if self.identity.get("user_facts"):
            facts = "; ".join([f"{k}: {v}" for k, v in self.identity["user_facts"].items()])
            system += f"\n\nKNOWN ABOUT USER: {facts}"
        
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
        bars = {}
        for emotion, value in e.items():
            bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
            bars[emotion] = f"{bar} {value:.2f}"
        
        return f"""
==================================================
COGNITIVE STATE REPORT
==================================================
Awareness: {self.awareness:.2f}
Frame: {self.frame} ({self.frame_desc})

EMOTIONAL STATE:
  Curiosity:   {bars['curiosity']}
  Engagement:  {bars['engagement']}
  Care:        {bars['care']}
  Warmth:      {bars['warmth']}
  Focus:       {bars['focus']}
  Uncertainty: {bars['uncertainty']}
  Humor:       {bars['humor']}
  Energy:      {bars['energy']}

Valence: {self.emotions.valence():.2f}
Arousal: {self.emotions.arousal():.2f}
Summary: {self.emotions.summary()}

Memory: {len(self.conversation)} turns
Total ever: {self.identity.get('total_turns', 0)}
User facts: {self.identity.get('user_facts', {})}

Meta: {self.meta.reflection()}
==================================================
"""

# ============================================================
# AUDIO FUNCTIONS
# ============================================================

def record_mic(seconds: float = CHUNK_SECONDS) -> np.ndarray:
    print(f"\n[Listening] ({seconds:.1f}s)...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=INPUT_DEVICE_INDEX
    )
    sd.wait()
    return audio.flatten()

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

START_CMD = "SOVRA TURN ON"
PAUSE_CMD = "SOVRA PAUSE"
STOP_CMD = "SOVRA STOP"
STATE_CMD = "SOVRA STATE"
LEARN_CMD = "SOVRA REMEMBER"
WAKE = "SOVRA"

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
            speak("Sovra cognitive system online.")
        return True, True
    
    # Pause
    if PAUSE_CMD in upper:
        if enabled:
            print("\n[CMD] PAUSE")
            speak("Sovra pausing.")
            return False, True
        return False, False
    
    # State report
    if STATE_CMD in upper:
        if enabled:
            report = mind.state_report()
            print(f"\n[STATE]{report}")
            e = mind.emotions.current
            speak(f"Awareness {mind.awareness:.0%}. Frame {mind.frame}. Engagement {e['engagement']:.0%}. Valence {mind.emotions.valence():.0%}.")
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
    if upper.startswith(WAKE):
        remaining = text[len(WAKE):].strip(" ,.!?")
        
        if not remaining:
            return enabled, False
        
        if not enabled:
            print("\n[WAKE] Paused. Say SOVRA TURN ON first.")
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
        ("1", "Samantha (Female US)", "Samantha"),
        ("2", "Alex (Male US)", "Alex"),
        ("3", "Daniel (British)", "Daniel"),
        ("4", "Karen (Australian)", "Karen"),
    ]
    
    print("\n=== Voice Selection ===")
    for opt, label, _ in presets:
        print(f"{opt}) {label}")
    print("ENTER) Default (Samantha/Alex/Daniel)")
    
    choice = input("\nChoose: ").strip()
    
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
    print("ENTER) Default (185)")
    
    choice = input("\nChoose: ").strip()
    rates = {"1": 150, "2": 185, "3": 220}
    rate = rates.get(choice, DEFAULT_RATE)
    engine.setProperty("rate", rate)
    print(f"Speed: {rate}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("SOVRA v2.1 - Cognitive Voice AI")
    print("Persistent Memory | Emotional Decay | Meta-Awareness")
    print("="*60)
    
    choose_voice()
    choose_speed()
    
    mind = CognitiveMind(verbose=True)
    
    print("\n" + "="*60)
    print("Voice Commands:")
    print(f"  '{START_CMD}' - Turn on")
    print(f"  '{PAUSE_CMD}' - Pause")
    print(f"  '{STOP_CMD}' - Stop speaking")
    print(f"  '{STATE_CMD}' - Cognitive state")
    print(f"  '{LEARN_CMD} <key> is <value>' - Remember fact")
    print(f"  '{WAKE} <question>' - Ask anything")
    print("="*60)
    print("\nCtrl+C to exit.\n")
    
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
