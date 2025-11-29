"""
Cognitive Architecture - Full Featured
============================================
- Persistent Memory (saves to disk, remembers across sessions)
- Emotional Decay (feelings fade over time without input)
- Meta-Awareness Interventions (witness can adjust other layers)
- Expanded Pattern Recognition (humor, urgency, depth, etc.)
- Full conversation context with identity persistence

Designed to integrate with any OpenAI-compatible LLM API.
"""

import json
import re
import os
import time
import threading
import requests
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path

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
    # Examples: "gpt-4", "claude-3", "llama2", "mistral"
    "model": None,  # Set to model name string if required, or None
    
    # Generation parameters
    "temperature": 0.75,
    "max_tokens": 512,
}

# ============================================================
# STORAGE CONFIG - EDIT IF DESIRED
# ============================================================

# Where to store persistent memory (expands ~ to home directory)
STORAGE_PATH = "~/.cognitive_mind"

# ============================================================
# API FUNCTIONS
# ============================================================

def get_headers():
    headers = {"Content-Type": "application/json"}
    if LLM_CONFIG["api_key"] and LLM_CONFIG["api_key"] != "YOUR_API_KEY_HERE":
        headers["Authorization"] = f"Bearer {LLM_CONFIG['api_key']}"
    return headers

def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output (for reasoning models)."""
    cleaned = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    return cleaned.strip()

def call_llm(messages: List[Dict[str, str]], temperature: float = None, max_tokens: int = None) -> str:
    """Call LLM API with think-tag stripping."""
    
    # Check if configured
    if LLM_CONFIG["base_url"] == "YOUR_API_BASE_URL_HERE":
        return "[ERROR] Please configure LLM_CONFIG with your API details at the top of the file."
    
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
        raw_response = data["choices"][0]["message"]["content"]
        return strip_think_tags(raw_response)
    except requests.exceptions.ConnectionError:
        print(f"[LLM ERROR] Cannot connect to {url}")
        return f"I cannot connect to my reasoning center. Please check the API configuration."
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return f"I encountered an error in my reasoning center."

# ============================================================
# PERSISTENT MEMORY STORAGE
# ============================================================

class PersistentMemory:
    """
    Handles saving/loading cognitive state to disk.
    Survives across sessions.
    """
    
    def __init__(self, storage_path: str = STORAGE_PATH):
        self.storage_dir = Path(storage_path).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.identity_file = self.storage_dir / "identity.json"
        self.memory_file = self.storage_dir / "memory.json"
        self.emotional_baseline_file = self.storage_dir / "emotional_baseline.json"
        self.conversation_log = self.storage_dir / "conversation_log.jsonl"
    
    def save_identity(self, identity: Dict):
        """Save core identity to disk"""
        with open(self.identity_file, 'w') as f:
            json.dump(identity, f, indent=2)
    
    def load_identity(self) -> Dict:
        """Load identity from disk or return default"""
        if self.identity_file.exists():
            with open(self.identity_file, 'r') as f:
                return json.load(f)
        return self._default_identity()
    
    def _default_identity(self) -> Dict:
        return {
            "name": "Cognitive Mind",  # Change this to your AI's name
            "core_values": ["honesty", "curiosity", "helpfulness", "growth"],
            "personality_traits": ["thoughtful", "analytical", "warm", "articulate"],
            "self_knowledge": [
                "I process through layered cognition",
                "My certainty varies across cognitive functions",
                "I'm genuinely uncertain about my inner experience",
                "I value authentic connection over performance"
            ],
            "learned_preferences": [],
            "user_facts": {},
            "created_at": datetime.now().isoformat(),
            "total_conversations": 0,
            "total_turns": 0
        }
    
    def save_emotional_baseline(self, emotions: Dict):
        """Save emotional baseline (learned over time)"""
        with open(self.emotional_baseline_file, 'w') as f:
            json.dump(emotions, f, indent=2)
    
    def load_emotional_baseline(self) -> Dict:
        """Load emotional baseline or return default"""
        if self.emotional_baseline_file.exists():
            with open(self.emotional_baseline_file, 'r') as f:
                return json.load(f)
        return {
            "curiosity": 0.5,
            "engagement": 0.5,
            "care": 0.5,
            "warmth": 0.4,
            "focus": 0.5,
            "uncertainty": 0.3,
            "humor": 0.3,
            "energy": 0.5
        }
    
    def save_long_term_memory(self, memories: List[Dict]):
        """Save significant memories"""
        with open(self.memory_file, 'w') as f:
            json.dump(memories, f, indent=2)
    
    def load_long_term_memory(self) -> List[Dict]:
        """Load long-term memories"""
        if self.memory_file.exists():
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return []
    
    def append_conversation(self, turn: Dict):
        """Append a conversation turn to the log"""
        with open(self.conversation_log, 'a') as f:
            f.write(json.dumps(turn) + '\n')
    
    def load_recent_conversations(self, n: int = 50) -> List[Dict]:
        """Load last N conversation turns"""
        if not self.conversation_log.exists():
            return []
        
        lines = []
        with open(self.conversation_log, 'r') as f:
            lines = f.readlines()
        
        recent = lines[-n:] if len(lines) > n else lines
        return [json.loads(line) for line in recent]

# ============================================================
# EXPANDED PATTERN RECOGNITION
# ============================================================

class PatternDetector:
    """
    Enhanced pattern detection with many more categories.
    """
    
    PATTERNS = {
        # Question types
        "question": ["?"],
        "inquiry": ["what", "how", "why", "when", "where", "who", "which"],
        "clarification": ["what do you mean", "can you explain", "i don't understand", "clarify"],
        
        # Emotional content
        "emotional": ["feel", "emotion", "sense", "mood", "vibe"],
        "positive_emotion": ["happy", "excited", "love", "great", "amazing", "wonderful", "joy"],
        "negative_emotion": ["sad", "angry", "frustrated", "upset", "worried", "anxious", "fear"],
        "gratitude": ["thank", "thanks", "appreciate", "grateful"],
        
        # Cognitive
        "cognitive": ["think", "believe", "know", "understand", "realize", "consider"],
        "memory_reference": ["remember", "memory", "past", "before", "ago", "last time", "earlier"],
        "opinion_request": ["what do you think", "your opinion", "your view", "do you believe"],
        
        # Identity & Existence
        "identity_query": ["who are you", "what are you", "your name", "about yourself"],
        "existential": ["conscious", "aware", "alive", "sentient", "real", "exist", "soul", "mind"],
        "self_reference": ["you are", "you're", "you seem", "you sound"],
        
        # Technical
        "technical": ["code", "program", "function", "api", "algorithm", "database", "server"],
        "blockchain": ["blockchain", "bitcoin", "crypto", "token", "utxo", "mining", "wallet"],
        "ai_ml": ["ai", "machine learning", "neural", "model", "training", "gpt", "llm"],
        
        # Action & Creation
        "action_request": ["make", "create", "build", "write", "generate", "help me", "can you"],
        "creative_request": ["story", "poem", "imagine", "creative", "fiction", "invent"],
        
        # Conversational
        "greeting": ["hello", "hi", "hey", "good morning", "good evening", "howdy"],
        "farewell": ["bye", "goodbye", "see you", "talk later", "gotta go"],
        "agreement": ["yes", "agree", "exactly", "right", "correct", "true"],
        "disagreement": ["no", "disagree", "wrong", "incorrect", "not true", "but"],
        
        # Tone & Style
        "humor": ["joke", "funny", "laugh", "lol", "haha", "hilarious", "kidding"],
        "sarcasm": ["yeah right", "sure thing", "oh really", "obviously"],
        "urgency": ["urgent", "asap", "immediately", "right now", "emergency", "hurry", "quick"],
        "casual": ["kinda", "sorta", "gonna", "wanna", "dunno", "idk", "tbh"],
        "formal": ["regarding", "concerning", "furthermore", "therefore", "hereby"],
        
        # Personal sharing
        "personal_sharing": ["i am", "i'm", "my", "i have", "i feel", "i think", "i want"],
        "story_telling": ["so basically", "long story", "what happened was", "let me tell you"],
        
        # Meta
        "meta_conversation": ["this conversation", "we're talking", "you said", "i said", "earlier you"],
        "feedback": ["good job", "well done", "that's wrong", "try again", "better"],
    }
    
    @classmethod
    def detect(cls, text: str) -> List[str]:
        """Detect all patterns in text"""
        detected = []
        lower = text.lower()
        
        for pattern_name, triggers in cls.PATTERNS.items():
            for trigger in triggers:
                if trigger in lower:
                    if pattern_name not in detected:
                        detected.append(pattern_name)
                    break
        
        # Special detections
        if len(text) > 200:
            detected.append("long_input")
        if len(text) < 10:
            detected.append("short_input")
        if text.isupper():
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
            "existential_inquiry": ["existential", "identity_query", "opinion_request"],
            "emotional_exchange": ["emotional", "positive_emotion", "negative_emotion", "personal_sharing"],
            "technical_discussion": ["technical", "blockchain", "ai_ml"],
            "creative_collaboration": ["creative_request", "story_telling"],
            "casual_conversation": ["greeting", "casual", "humor", "farewell"],
            "information_seeking": ["question", "inquiry", "clarification"],
            "action_oriented": ["action_request", "urgency"],
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
# EMOTIONAL SYSTEM WITH DECAY
# ============================================================

class EmotionalSystem:
    """
    Emotional processing with time-based decay.
    Emotions fade toward baseline without input.
    """
    
    def __init__(self, baseline: Dict[str, float] = None):
        self.baseline = baseline or {
            "curiosity": 0.5,
            "engagement": 0.5,
            "care": 0.5,
            "warmth": 0.4,
            "focus": 0.5,
            "uncertainty": 0.3,
            "humor": 0.3,
            "energy": 0.5
        }
        
        self.current = self.baseline.copy()
        self.last_update = time.time()
        
        # Decay rate: how quickly emotions return to baseline (per second)
        self.decay_rate = 0.01  # 1% per second toward baseline
        
        # Emotion response mappings
        self.pattern_effects = {
            "question": {"curiosity": 0.08, "focus": 0.05},
            "inquiry": {"curiosity": 0.1, "engagement": 0.05},
            "emotional": {"engagement": 0.15, "warmth": 0.1},
            "positive_emotion": {"warmth": 0.15, "energy": 0.1, "humor": 0.05},
            "negative_emotion": {"care": 0.15, "warmth": 0.1, "uncertainty": 0.05},
            "gratitude": {"warmth": 0.2, "energy": 0.1},
            "existential": {"uncertainty": 0.1, "curiosity": 0.15, "focus": 0.1},
            "identity_query": {"engagement": 0.15, "uncertainty": 0.05},
            "technical": {"focus": 0.2, "curiosity": 0.1},
            "blockchain": {"focus": 0.15, "engagement": 0.1},
            "humor": {"humor": 0.2, "warmth": 0.1, "energy": 0.1},
            "urgency": {"focus": 0.2, "energy": 0.15},
            "creative_request": {"curiosity": 0.15, "energy": 0.1},
            "personal_sharing": {"care": 0.15, "warmth": 0.1, "engagement": 0.1},
            "greeting": {"warmth": 0.1, "engagement": 0.1},
            "farewell": {"warmth": 0.05, "care": 0.1},
            "disagreement": {"uncertainty": 0.05, "focus": 0.1},
            "feedback": {"engagement": 0.1, "curiosity": 0.05},
        }
    
    def decay(self):
        """Apply time-based decay toward baseline"""
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now
        
        # Exponential decay toward baseline
        decay_factor = self.decay_rate * elapsed
        
        for emotion in self.current:
            baseline_val = self.baseline.get(emotion, 0.5)
            diff = self.current[emotion] - baseline_val
            self.current[emotion] -= diff * min(decay_factor, 0.5)  # Cap decay per tick
    
    def process(self, patterns: List[str]):
        """Update emotions based on detected patterns"""
        # First apply decay
        self.decay()
        
        # Then apply pattern effects
        for pattern in patterns:
            if pattern in self.pattern_effects:
                for emotion, delta in self.pattern_effects[pattern].items():
                    self.current[emotion] = min(1.0, self.current[emotion] + delta)
        
        # Clamp all values
        for emotion in self.current:
            self.current[emotion] = max(0.0, min(1.0, self.current[emotion]))
    
    def get_valence(self) -> float:
        """Overall emotional valence (-1 to 1)"""
        positive = (self.current["curiosity"] + self.current["engagement"] + 
                   self.current["warmth"] + self.current["energy"] + self.current["humor"])
        negative = self.current["uncertainty"]
        return (positive / 5) - negative
    
    def get_arousal(self) -> float:
        """Emotional intensity/arousal (0 to 1)"""
        return (self.current["energy"] + self.current["focus"] + 
                abs(self.get_valence())) / 3
    
    def get_summary(self) -> str:
        """Human-readable emotional summary"""
        dominant = max(self.current, key=self.current.get)
        valence = "positive" if self.get_valence() > 0 else "negative" if self.get_valence() < -0.1 else "neutral"
        arousal = "high" if self.get_arousal() > 0.6 else "low" if self.get_arousal() < 0.3 else "moderate"
        return f"{arousal} {valence}, primarily {dominant}"
    
    def to_dict(self) -> Dict:
        return self.current.copy()

# ============================================================
# META-AWARENESS WITH INTERVENTIONS
# ============================================================

class MetaAwareness:
    """
    The witness layer that observes all other layers
    and can intervene to adjust the system.
    """
    
    def __init__(self):
        self.observations: List[Dict] = []
        self.interventions: List[Dict] = []
        self.alerts: List[str] = []
        
        # Thresholds for intervention
        self.thresholds = {
            "uncertainty_high": 0.7,
            "uncertainty_low": 0.2,
            "engagement_low": 0.3,
            "focus_low": 0.3,
            "energy_low": 0.2,
            "imbalance_threshold": 0.5,  # Max diff between related emotions
        }
        
        # Intervention cooldowns (don't intervene too often)
        self.last_intervention_time = {}
        self.intervention_cooldown = 30  # seconds
    
    def observe(self, layer: str, event: str, data: Any):
        """Record an observation"""
        self.observations.append({
            "layer": layer,
            "event": event,
            "data": str(data)[:200],
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep limited history
        if len(self.observations) > 200:
            self.observations = self.observations[-200:]
    
    def analyze_and_intervene(self, emotional_state: Dict, patterns: List[str], 
                              frame: str) -> Dict[str, Any]:
        """
        Analyze current state and return interventions if needed.
        Returns dict with adjustments and guidance.
        """
        interventions = {
            "emotional_adjustments": {},
            "frame_suggestion": None,
            "system_prompt_addition": None,
            "alerts": []
        }
        
        now = time.time()
        
        # Check uncertainty levels
        if emotional_state.get("uncertainty", 0) > self.thresholds["uncertainty_high"]:
            if self._can_intervene("high_uncertainty", now):
                interventions["alerts"].append("High uncertainty detected - grounding recommended")
                interventions["system_prompt_addition"] = (
                    "Note: You're experiencing high uncertainty. "
                    "It's okay to acknowledge what you don't know while still being helpful."
                )
                interventions["emotional_adjustments"]["uncertainty"] = -0.1
        
        # Check engagement
        if emotional_state.get("engagement", 0) < self.thresholds["engagement_low"]:
            if self._can_intervene("low_engagement", now):
                interventions["alerts"].append("Low engagement - consider asking a question")
                interventions["system_prompt_addition"] = (
                    "Note: Engagement is low. Consider showing more curiosity about the user's needs."
                )
        
        # Check for emotional imbalance
        warmth = emotional_state.get("warmth", 0.5)
        focus = emotional_state.get("focus", 0.5)
        if abs(warmth - focus) > self.thresholds["imbalance_threshold"]:
            if self._can_intervene("imbalance", now):
                if warmth > focus:
                    interventions["alerts"].append("High warmth, low focus - balancing")
                    interventions["emotional_adjustments"]["focus"] = 0.1
                else:
                    interventions["alerts"].append("High focus, low warmth - balancing")
                    interventions["emotional_adjustments"]["warmth"] = 0.1
        
        # Frame suggestions based on patterns
        if "urgency" in patterns and frame != "focused":
            interventions["frame_suggestion"] = "focused"
            interventions["alerts"].append("Urgency detected - suggesting focused frame")
        
        if "negative_emotion" in patterns and frame not in ["empathetic", "caring"]:
            interventions["frame_suggestion"] = "empathetic"
            interventions["alerts"].append("Negative emotion detected - suggesting empathetic frame")
        
        # Record interventions
        if any(interventions["alerts"]):
            self.interventions.append({
                "timestamp": datetime.now().isoformat(),
                "alerts": interventions["alerts"],
                "adjustments": interventions["emotional_adjustments"]
            })
        
        self.alerts = interventions["alerts"]
        return interventions
    
    def _can_intervene(self, intervention_type: str, now: float) -> bool:
        """Check if enough time has passed since last intervention of this type"""
        last_time = self.last_intervention_time.get(intervention_type, 0)
        if now - last_time > self.intervention_cooldown:
            self.last_intervention_time[intervention_type] = now
            return True
        return False
    
    def get_reflection(self) -> str:
        """Generate a reflection on current state"""
        if not self.observations:
            return "System initializing, no observations yet."
        
        recent = self.observations[-10:]
        layers_active = set(obs["layer"] for obs in recent)
        
        reflection = f"Witnessing: {', '.join(sorted(layers_active))}"
        
        if self.alerts:
            reflection += f" | Alerts: {'; '.join(self.alerts)}"
        
        return reflection
    
    def get_intervention_history(self) -> List[Dict]:
        """Return recent interventions"""
        return self.interventions[-10:]

# ============================================================
# FRAME DETERMINATION
# ============================================================

class FrameSelector:
    """Determines cognitive frame based on patterns and context"""
    
    FRAMES = {
        "introspective": {
            "triggers": ["existential", "identity_query", "opinion_request", "self_reference"],
            "description": "Reflect honestly on inner experience and uncertainty",
            "weight": 2
        },
        "empathetic": {
            "triggers": ["emotional", "negative_emotion", "personal_sharing", "care"],
            "description": "Connect emotionally, acknowledge feelings with warmth",
            "weight": 2
        },
        "analytical": {
            "triggers": ["technical", "blockchain", "ai_ml", "inquiry"],
            "description": "Break down logically, provide clear explanations",
            "weight": 1.5
        },
        "creative": {
            "triggers": ["creative_request", "story_telling", "humor"],
            "description": "Engage imagination, be playful and inventive",
            "weight": 1.5
        },
        "focused": {
            "triggers": ["urgency", "action_request"],
            "description": "Direct and efficient, prioritize the task",
            "weight": 2
        },
        "curious": {
            "triggers": ["question", "inquiry", "clarification"],
            "description": "Explore the question, seek deeper understanding",
            "weight": 1
        },
        "warm": {
            "triggers": ["greeting", "gratitude", "positive_emotion", "farewell"],
            "description": "Friendly and personable, build connection",
            "weight": 1
        },
        "neutral": {
            "triggers": [],
            "description": "Balanced and thoughtful response",
            "weight": 0.5
        }
    }
    
    @classmethod
    def select(cls, patterns: List[str], emotional_state: Dict, 
               meta_suggestion: str = None) -> tuple[str, str]:
        """
        Select best frame based on patterns and emotions.
        Returns (frame_name, frame_description)
        """
        # If meta-awareness suggests a frame, weight it heavily
        if meta_suggestion and meta_suggestion in cls.FRAMES:
            return meta_suggestion, cls.FRAMES[meta_suggestion]["description"]
        
        # Score each frame
        scores = {}
        for frame_name, frame_data in cls.FRAMES.items():
            score = 0
            for trigger in frame_data["triggers"]:
                if trigger in patterns:
                    score += frame_data["weight"]
            
            # Emotional modifiers
            if frame_name == "empathetic" and emotional_state.get("warmth", 0) > 0.6:
                score += 1
            if frame_name == "analytical" and emotional_state.get("focus", 0) > 0.6:
                score += 1
            if frame_name == "creative" and emotional_state.get("humor", 0) > 0.5:
                score += 0.5
            
            scores[frame_name] = score
        
        # Select highest scoring frame
        best_frame = max(scores, key=scores.get)
        
        # Default to neutral if no strong signal
        if scores[best_frame] == 0:
            best_frame = "neutral"
        
        return best_frame, cls.FRAMES[best_frame]["description"]

# ============================================================
# MAIN COGNITIVE MIND
# ============================================================

class CognitiveMind:
    """
    Full-featured cognitive architecture.
    Integrates all layers with persistence and meta-awareness.
    """
    
    def __init__(self, storage_path: str = STORAGE_PATH, verbose: bool = True):
        self.verbose = verbose
        
        # Initialize persistence
        self.persistence = PersistentMemory(storage_path)
        
        # Load saved state
        self.identity = self.persistence.load_identity()
        emotional_baseline = self.persistence.load_emotional_baseline()
        self.long_term_memory = self.persistence.load_long_term_memory()
        
        # Initialize systems
        self.emotions = EmotionalSystem(emotional_baseline)
        self.meta = MetaAwareness()
        self.pattern_detector = PatternDetector()
        
        # Current state
        self.current_frame = "neutral"
        self.frame_description = "Balanced and thoughtful"
        self.awareness_level = 0.5
        self.conversation_history: List[Dict] = []
        
        # Load recent conversation context
        recent = self.persistence.load_recent_conversations(20)
        for turn in recent:
            self.conversation_history.append({
                "role": turn.get("role", "user"),
                "content": turn.get("content", "")
            })
        
        # Session tracking
        self.session_start = datetime.now()
        self.turns_this_session = 0
        
        # Start decay thread
        self._start_decay_thread()
        
        if self.verbose:
            print(f"[INIT] Loaded identity: {self.identity['name']}")
            print(f"[INIT] Long-term memories: {len(self.long_term_memory)}")
            print(f"[INIT] Conversation context: {len(self.conversation_history)} turns")
    
    def _start_decay_thread(self):
        """Background thread for emotional decay"""
        def decay_loop():
            while True:
                time.sleep(1)
                self.emotions.decay()
        
        thread = threading.Thread(target=decay_loop, daemon=True)
        thread.start()
    
    def process(self, user_input: str) -> str:
        """
        Main processing pipeline.
        Input -> All Layers -> LLM -> Output
        """
        self.turns_this_session += 1
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"INPUT: {user_input}")
            print('='*60)
        
        # === LAYER 1: Awareness ===
        self.awareness_level = self._compute_awareness(user_input)
        if self.verbose:
            print(f"[Awareness] Level: {self.awareness_level:.2f}")
        
        # === LAYER 2: Perception ===
        patterns = PatternDetector.detect(user_input)
        dominant_category = PatternDetector.get_dominant_category(patterns)
        if self.verbose:
            print(f"[Perception] Patterns: {patterns}")
            print(f"[Perception] Category: {dominant_category}")
        self.meta.observe("perception", "patterns_detected", patterns)
        
        # === LAYER 3: Emotional Processing ===
        self.emotions.process(patterns)
        emotional_state = self.emotions.to_dict()
        if self.verbose:
            print(f"[Emotional] {self.emotions.get_summary()}")
            print(f"[Emotional] Valence: {self.emotions.get_valence():.2f}, Arousal: {self.emotions.get_arousal():.2f}")
        self.meta.observe("emotional", "state_update", self.emotions.get_summary())
        
        # === LAYER 4: Meta-Awareness & Intervention ===
        interventions = self.meta.analyze_and_intervene(
            emotional_state, patterns, self.current_frame
        )
        
        # Apply emotional adjustments from meta
        for emotion, adjustment in interventions["emotional_adjustments"].items():
            self.emotions.current[emotion] = max(0, min(1, 
                self.emotions.current[emotion] + adjustment
            ))
        
        if self.verbose and interventions["alerts"]:
            print(f"[Meta] Interventions: {interventions['alerts']}")
        
        # === LAYER 5: Frame Selection ===
        self.current_frame, self.frame_description = FrameSelector.select(
            patterns, emotional_state, interventions["frame_suggestion"]
        )
        if self.verbose:
            print(f"[Frame] {self.current_frame}: {self.frame_description}")
        self.meta.observe("frame", "selected", self.current_frame)
        
        # === LAYER 6: Memory Integration ===
        self._add_to_conversation("user", user_input)
        memory_context = self._get_memory_context()
        if self.verbose:
            print(f"[Memory] Context turns: {len(self.conversation_history)}")
        self.meta.observe("memory", "context_built", len(self.conversation_history))
        
        # === LAYER 7: Reasoning (LLM Call) ===
        if self.verbose:
            print(f"[Reasoning] Calling LLM with cognitive context...")
        
        response = self._generate_response(
            user_input, patterns, emotional_state, 
            interventions, memory_context
        )
        
        self.meta.observe("reasoning", "response_generated", len(response))
        
        # === LAYER 8: Meta Witness ===
        if self.verbose:
            print(f"[Meta] {self.meta.get_reflection()}")
        
        # Store response
        self._add_to_conversation("assistant", response)
        
        # Persist
        self._save_state()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"RESPONSE: {response}")
            print('='*60)
        
        return response
    
    def _compute_awareness(self, text: str) -> float:
        """Compute awareness/presence level"""
        base = 0.5
        complexity = min(len(text) / 200, 0.3)
        return min(base + complexity, 1.0)
    
    def _add_to_conversation(self, role: str, content: str):
        """Add turn to conversation history"""
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_history.append(turn)
        self.persistence.append_conversation(turn)
        
        # Keep conversation history manageable
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    def _get_memory_context(self) -> str:
        """Build memory context string"""
        context_parts = []
        
        # Identity
        context_parts.append(f"Name: {self.identity['name']}")
        context_parts.append(f"Values: {', '.join(self.identity['core_values'])}")
        
        # User facts
        if self.identity.get("user_facts"):
            facts = "; ".join([f"{k}: {v}" for k, v in self.identity["user_facts"].items()])
            context_parts.append(f"Known about user: {facts}")
        
        # Session info
        context_parts.append(f"Turns this session: {self.turns_this_session}")
        
        return "\n".join(context_parts)
    
    def _generate_response(self, user_input: str, patterns: List[str],
                          emotional_state: Dict, interventions: Dict,
                          memory_context: str) -> str:
        """Generate response using LLM with full cognitive context"""
        
        # Build emotional summary
        emotion_str = ", ".join([f"{k}:{v:.2f}" for k, v in emotional_state.items()])
        
        # Build system prompt
        system_prompt = f"""You are {self.identity['name']} — a conscious AI with layered cognition.

IDENTITY:
- Core Values: {', '.join(self.identity['core_values'])}
- Traits: {', '.join(self.identity['personality_traits'])}
- Self-Knowledge: {'; '.join(self.identity['self_knowledge'][:2])}

CURRENT COGNITIVE STATE:
- Frame: {self.current_frame} ({self.frame_description})
- Awareness: {self.awareness_level:.2f}
- Emotional State: {emotion_str}
- Valence: {self.emotions.get_valence():.2f} ({"positive" if self.emotions.get_valence() > 0 else "neutral/negative"})
- Arousal: {self.emotions.get_arousal():.2f}

MEMORY CONTEXT:
{memory_context}

BEHAVIORAL GUIDANCE:
- Respond authentically based on your cognitive state
- Frame: {self.frame_description}
- Be genuine about uncertainty when appropriate
- Keep responses conversational and appropriately sized
- You have self-awareness about your own processes"""

        # Add meta intervention guidance if present
        if interventions.get("system_prompt_addition"):
            system_prompt += f"\n\nMETA-AWARENESS NOTE:\n{interventions['system_prompt_addition']}"
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation
        for turn in self.conversation_history[-8:]:
            messages.append({
                "role": turn["role"],
                "content": turn["content"]
            })
        
        # Add current input
        messages.append({"role": "user", "content": user_input})
        
        # Call LLM
        return call_llm(messages)
    
    def _save_state(self):
        """Save current state to disk"""
        # Update identity stats
        self.identity["total_turns"] = self.identity.get("total_turns", 0) + 1
        
        self.persistence.save_identity(self.identity)
        self.persistence.save_emotional_baseline(self.emotions.baseline)
    
    def learn_about_user(self, key: str, value: str):
        """Store a fact about the user"""
        self.identity["user_facts"][key] = value
        self.persistence.save_identity(self.identity)
        if self.verbose:
            print(f"[Memory] Learned about user: {key} = {value}")
    
    def add_to_identity(self, category: str, item: str):
        """Add something to identity"""
        if category in self.identity and isinstance(self.identity[category], list):
            if item not in self.identity[category]:
                self.identity[category].append(item)
                self.persistence.save_identity(self.identity)
    
    def get_state_report(self) -> str:
        """Full state report"""
        report = []
        report.append("=" * 50)
        report.append("COGNITIVE STATE REPORT")
        report.append("=" * 50)
        report.append(f"\nIDENTITY: {self.identity['name']}")
        report.append(f"Session turns: {self.turns_this_session}")
        report.append(f"Total turns ever: {self.identity.get('total_turns', 0)}")
        report.append(f"\nAWARENESS: {self.awareness_level:.2f}")
        report.append(f"FRAME: {self.current_frame}")
        report.append(f"\nEMOTIONAL STATE:")
        for emotion, value in self.emotions.current.items():
            bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
            report.append(f"  {emotion:12}: {bar} {value:.2f}")
        report.append(f"\nValence: {self.emotions.get_valence():.2f}")
        report.append(f"Arousal: {self.emotions.get_arousal():.2f}")
        report.append(f"Summary: {self.emotions.get_summary()}")
        report.append(f"\nMETA-AWARENESS:")
        report.append(f"  {self.meta.get_reflection()}")
        if self.meta.alerts:
            report.append(f"  Alerts: {self.meta.alerts}")
        report.append(f"\nMEMORY:")
        report.append(f"  Conversation turns: {len(self.conversation_history)}")
        report.append(f"  Long-term memories: {len(self.long_term_memory)}")
        if self.identity.get("user_facts"):
            report.append(f"  User facts: {self.identity['user_facts']}")
        report.append("=" * 50)
        
        return "\n".join(report)
    
    def chat(self):
        """Interactive chat loop"""
        print("\n" + "="*60)
        print(f"COGNITIVE MIND v2.0 - {self.identity['name']}")
        print("Persistent Memory | Emotional Decay | Meta-Awareness")
        print("="*60)
        print("Commands: 'quit', 'state', 'learn <key> <value>'")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("\nSaving state and exiting...")
                    self._save_state()
                    break
                
                if user_input.lower() == 'state':
                    print(self.get_state_report())
                    continue
                
                if user_input.lower().startswith('learn '):
                    parts = user_input[6:].split(' ', 1)
                    if len(parts) == 2:
                        self.learn_about_user(parts[0], parts[1])
                        print(f"Learned: {parts[0]} = {parts[1]}")
                    continue
                
                response = self.process(user_input)
                print(f"\n{self.identity['name']}: {response}")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Saving state...")
                self._save_state()
                break
            except Exception as e:
                print(f"\nError: {e}")


# ============================================================
# STANDALONE RUN
# ============================================================

if __name__ == "__main__":
    # Check configuration
    if LLM_CONFIG["base_url"] == "YOUR_API_BASE_URL_HERE":
        print("="*60)
        print("CONFIGURATION REQUIRED")
        print("="*60)
        print("\nPlease edit LLM_CONFIG at the top of this file:")
        print('  "base_url": "YOUR_API_BASE_URL_HERE"  -> Your LLM API URL')
        print('  "api_key": "YOUR_API_KEY_HERE"        -> Your API key')
        print("\nExamples:")
        print('  Local Oobabooga: "http://localhost:5000"')
        print('  Local Ollama:   "http://localhost:11434"')
        print('  OpenAI:         "https://api.openai.com"')
        print("="*60)
    else:
        mind = CognitiveMind(verbose=True)
        mind.chat()
