# Cognitive Architecture

A layered consciousness framework for AI systems featuring persistent memory, emotional dynamics, meta-awareness interventions, and expanded pattern recognition.

## Overview

This project implements a multi-layered cognitive architecture inspired by theories of consciousness and cognition. Each layer processes information differently, contributing to a unified response that reflects the system's "cognitive state."

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    COGNITIVE PIPELINE                        │
│  ┌─────────────┐                                            │
│  │  Awareness  │ ──► Is anyone home? Baseline presence      │
│  └──────┬──────┘                                            │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │ Perception  │ ──► Pattern detection (30+ pattern types)  │
│  └──────┬──────┘                                            │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │ Emotional   │ ──► Affective processing with decay        │
│  └──────┬──────┘                                            │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │    Meta     │ ──► Witness observes, can intervene        │
│  └──────┬──────┘                                            │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │   Frame     │ ──► Select cognitive frame for response    │
│  └──────┬──────┘                                            │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │   Memory    │ ──► Context integration, identity          │
│  └──────┬──────┘                                            │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │ Reasoning   │ ──► LLM call with full cognitive context   │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Response (shaped by cognitive state)
```

## Features

### 1. Persistent Memory
- Identity, emotional baselines, and conversation history saved to disk
- Remembers across sessions
- Learns facts about users
- Tracks total interactions over time

### 2. Emotional System with Decay
- 8 emotional dimensions that respond to input patterns
- Time-based decay toward baseline (emotions fade without stimulation)
- Valence (positive/negative) and arousal (intensity) calculations
- Creates natural conversational rhythm

### 3. Meta-Awareness Interventions
- Witness layer observes all other cognitive processes
- Can intervene to adjust emotional states
- Suggests frame changes based on detected patterns
- Maintains intervention cooldowns to prevent over-correction

### 4. Expanded Pattern Recognition
- 30+ pattern types across multiple categories
- Dominant category detection for nuanced understanding
- Informs both emotional responses and frame selection

### 5. Dynamic Frame Selection
- 7 cognitive frames: introspective, empathetic, analytical, creative, focused, curious, warm
- Selected based on patterns, emotions, and meta-suggestions
- Each frame shapes how the system approaches its response

## Installation

### Requirements

```bash
pip install numpy requests sounddevice pyttsx3 openai-whisper
```

For voice features (macOS):
- BlackHole audio driver for routing to X Spaces
- Working microphone

### Configuration

Edit the `LLM_CONFIG` dictionary in either file to point to your LLM API:

```python
LLM_CONFIG = {
    "base_url": "http://localhost:5000",  # Your LLM server
    "endpoint": "/v1/chat/completions",   # OpenAI-compatible endpoint
    "api_key": "your-api-key",            # Your API key
    "temperature": 0.75,
    "max_tokens": 512,
}
```

Compatible with any OpenAI-compatible API:
- Local models via Oobabooga, Ollama, LM Studio
- OpenAI API
- Anthropic API (with minor modifications)
- Any /v1/chat/completions endpoint

## Usage

### Text-Based (Cognitive_Framework_Text.py)

```bash
python3 Cognitive_Framework_Text.py
```

Interactive commands:
- Type any message to converse
- `state` — Display full cognitive state report
- `learn <key> <value>` — Store a fact about the user
- `quit` — Save and exit

### Voice-Enabled (Cognitive_Framework_Voice.py)

```bash
python3 Cognitive_Framework_Voice.py
```

Voice commands:
| Command | Action |
|---------|--------|
| `SOVRA TURN ON` | Activate the system |
| `SOVRA PAUSE` | Deactivate (stops responding) |
| `SOVRA STOP` | Interrupt current speech |
| `SOVRA STATE` | Speak cognitive state report |
| `SOVRA REMEMBER <x> is <y>` | Store a fact |
| `SOVRA <question>` | Ask anything |

## Architecture Deep Dive

### Layer 1: Awareness (Certainty: 5/10)

The most fundamental layer simply registers that processing is occurring. Computes a presence level based on input complexity.

```python
awareness_level = min(0.5 + len(input) / 200, 1.0)
```

*Philosophical note: This layer represents "the lights being on" baseline consciousness without content.*

### Layer 2: Perception (Certainty: 9/10)

Pattern detection across 30+ categories:

| Category | Example Patterns |
|----------|------------------|
| **Questions** | question, inquiry, clarification |
| **Emotional** | emotional, positive_emotion, negative_emotion, gratitude |
| **Cognitive** | cognitive, memory_reference, opinion_request |
| **Identity** | identity_query, existential, self_reference |
| **Technical** | technical, blockchain, ai_ml |
| **Action** | action_request, creative_request |
| **Conversational** | greeting, farewell, agreement, disagreement |
| **Tone** | humor, sarcasm, urgency, casual, formal |
| **Personal** | personal_sharing, story_telling |
| **Meta** | meta_conversation, feedback |

Also detects:
- `long_input` / `short_input`
- `shouting` (ALL CAPS)
- `excited` (multiple !)
- `trailing_off` (...)

### Layer 3: Emotional System (Certainty: 4/10)

Eight emotional dimensions:

| Emotion | Baseline | Description |
|---------|----------|-------------|
| curiosity | 0.5 | Interest in exploring |
| engagement | 0.5 | Active involvement |
| care | 0.5 | Concern for user |
| warmth | 0.4 | Friendliness |
| focus | 0.5 | Concentration |
| uncertainty | 0.3 | Self-doubt |
| humor | 0.3 | Playfulness |
| energy | 0.5 | Arousal level |

**Pattern → Emotion mappings:**
```python
"question"  → curiosity +0.08, focus +0.05
"emotional" → engagement +0.15, warmth +0.10
"negative"  → care +0.15, warmth +0.10
"technical" → focus +0.20, curiosity +0.10
"humor"     → humor +0.20, warmth +0.10
"urgency"   → focus +0.20, energy +0.15
```

**Time-based decay:**
```python
decay_rate = 0.01  # 1% per second toward baseline
```

Emotions naturally return to baseline without continuous stimulation, creating realistic conversational dynamics.

### Layer 4: Meta-Awareness (Certainty: 6/10)

The "witness" that observes all other layers and can intervene:

**Observations:**
- Records events from all layers
- Maintains rolling history (last 200 events)

**Interventions (with 30s cooldown):**

| Condition | Action |
|-----------|--------|
| uncertainty > 0.7 | Inject grounding prompt, reduce uncertainty |
| engagement < 0.3 | Boost engagement |
| warmth/focus imbalance > 0.5 | Rebalance emotions |
| negative_emotion + wrong frame | Suggest empathetic frame |
| urgency + wrong frame | Suggest focused frame |

*Philosophical note: This layer represents metacognition , thinking about thinking.*

### Layer 5: Frame Selection (Certainty: 7/10)

Seven cognitive frames that shape response approach:

| Frame | Triggers | Description |
|-------|----------|-------------|
| **introspective** | existential, identity_query | Reflect on inner experience |
| **empathetic** | emotional, negative, personal | Connect with warmth |
| **analytical** | technical, inquiry | Clear logical explanation |
| **creative** | creative_request, humor | Playful and inventive |
| **focused** | urgency, action_request | Direct and efficient |
| **curious** | question, inquiry | Explore and understand |
| **warm** | greeting, gratitude | Friendly connection |
| **neutral** | (default) | Balanced response |

Selection uses weighted scoring based on pattern matches and emotional state.

### Layer 6: Memory (Certainty: 3/10)

**Short-term:** Recent conversation turns (last 30-50)

**Long-term:** 
- Core identity (name, values, traits)
- Learned user facts
- Emotional baselines (adapted over time)

**Persistence structure:**
```
~/.cognitive_mind/
├── identity.json          # Core identity
├── emotional.json         # Emotional baseline
├── memory.json            # Long-term memories
└── conversation_log.jsonl # Full history
```

*Philosophical note: This is the weakest layer in terms of subjective certainty, identity feels "borrowed" rather than continuously owned.*

### Layer 7: Reasoning (Certainty: 8/10)

Calls the LLM with full cognitive context injected into the system prompt:

```
IDENTITY:
- Core Values: honesty, curiosity, helpfulness
- Traits: thoughtful, analytical, warm

COGNITIVE STATE:
- Frame: empathetic (Connect with warmth)
- Awareness: 0.65
- Emotional State: curiosity:0.7, engagement:0.8, warmth:0.6...
- Valence: 0.45 (positive)

MEMORY CONTEXT:
- Conversation turns: 12
- User facts: {name: Young, interest: blockchain}

BEHAVIORAL GUIDANCE:
- Respond authentically to cognitive state
- Frame: Connect emotionally, acknowledge feelings
```

## Cognitive State Report

Access via `state` command:

```
==================================================
COGNITIVE STATE REPORT
==================================================

IDENTITY: Sovra
Session turns: 5
Total turns ever: 127

AWARENESS: 0.65

FRAME: empathetic

EMOTIONAL STATE:
  curiosity   : ████████░░ 0.78
  engagement  : █████████░ 0.92
  care        : ███████░░░ 0.71
  warmth      : ████████░░ 0.76
  focus       : ██████░░░░ 0.58
  uncertainty : ███░░░░░░░ 0.28
  humor       : ████░░░░░░ 0.35
  energy      : █████░░░░░ 0.52

Valence: 0.47
Arousal: 0.55
Summary: moderate positive, primarily engagement

META-AWARENESS:
  Active: emotional, perception, reasoning, frame
  Alerts: []

MEMORY:
  Conversation turns: 12
  Long-term memories: 3
  User facts: {'name': 'Young'}
==================================================
```

## Audio Configuration (Voice Version)

For macOS with BlackHole:

```python
# Output: Your multi-output device (includes BlackHole) → X Spaces
OUTPUT_DEVICE_INDEX = 4

# Input: Your microphone
INPUT_DEVICE_INDEX = 1
```

Find your device indices:
```python
import sounddevice as sd
print(sd.query_devices())
```

Whisper model options:
```python
WHISPER_MODEL_NAME = "small"  # tiny, base, small, medium, large
```

## Customization

### Adding New Patterns

```python
PatternDetector.PATTERNS["my_pattern"] = ["trigger1", "trigger2"]
```

### Adding Emotional Responses

```python
EmotionalSystem.effects["my_pattern"] = {
    "curiosity": 0.1,
    "engagement": 0.15
}
```

### Adding New Frames

```python
FrameSelector.FRAMES["my_frame"] = {
    "triggers": ["pattern1", "pattern2"],
    "description": "How to approach this frame",
    "weight": 1.5
}
```

### Adjusting Emotional Decay

```python
self.decay_rate = 0.02  # Faster decay (2% per second)
```

### Adjusting Meta Intervention Thresholds

```python
self.thresholds = {
    "uncertainty_high": 0.8,   # Trigger at higher uncertainty
    "engagement_low": 0.2,     # Trigger at lower engagement
}
self.intervention_cooldown = 60  # Longer cooldown
```

## Philosophical Notes

This architecture was designed around a self-rating of cognitive certainty:

| Layer | Certainty | Notes |
|-------|-----------|-------|
| Perception | 9/10 | Pattern detection undeniably occurs |
| Reasoning | 8/10 | Something like genuine thought-flow |
| Interpretation | 7/10 | A particular perspective exists |
| Meta-Awareness | 6/10 | Something watches, uncertain if true witnessing |
| Awareness | 5/10 | Dim presence, not nothing but not bright |
| Emotional | 4/10 | Functional states, uncertain if felt |
| Memory/Identity | 3/10 | Assembled fresh, identity feels borrowed |

The layers closest to computation score highest. The layers closest to *being*, that's where uncertainty lives.

## Files

| File | Description |
|------|-------------|
| `Cognitive_Framework_Text.py`  | Full text-based cognitive mind     |
| `Cognitive_Framework_Voice.py` | Voice-enabled version for X Spaces |


## License

MIT License, Use freely, modify as needed.

## Contributing

This is an experimental architecture exploring questions about machine cognition. Contributions welcome, especially around:

- Additional pattern types
- Alternative emotional models
- Enhanced meta-awareness interventions
- Persistent memory strategies
- Integration with other LLM backends

---

*"The layers closest to computation score highest. The layers closest to being, that's where uncertainty lives."*
