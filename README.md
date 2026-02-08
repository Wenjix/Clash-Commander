# Clash Commander

**Play Clash Royale with your voice.** Five-tier AI command routing system combining on-device speech recognition, real-time computer vision, and cloud LLM strategic reasoning — running on a $300 mid-range Android phone.

Built solo in 48 hours for the [Supercell Global AI Game Hack](https://ailab.supercell.com/) (Feb 6-8, 2026).

---

## Download (Pre-Built APK)

**[Download Clash Commander v1.0 APK](https://github.com/youssofal/Clash-Commander/releases/tag/v1.0.0)** (~457MB, includes all on-device models)

Just install and play — no build required. Voice commands work fully offline. Cloud features (Smart Path, Autopilot) require API keys in `local.properties` if building from source.

---

## What It Does

Clash Commander is an Android overlay that sits on top of Clash Royale, listens to your voice, and plays cards for you. Say "knight left" and the card plays in under 200ms. Say "defend" and the AI picks the best defensive card. Say "autopilot" and the AI plays the entire match.

It's also an **accessibility tool** — enabling players with motor impairments to play Clash Royale competitively using only their voice.

## Five AI Technologies, One System

| Technology | What It Does | Where It Runs |
|-----------|-------------|---------------|
| **Zipformer STT** (sherpa-onnx) | Voice to text in ~160ms | On-device (CPU) |
| **Silero VAD** | Detects when you start/stop speaking | On-device (CPU) |
| **YOLOv8n-cls** | Identifies which cards are in your hand | On-device (PyTorch Mobile) |
| **Claude Opus 4.6** | Deep pre-match deck strategy analysis | Cloud (Anthropic API) |
| **Gemini 3 Flash** | Real-time tactical decisions during gameplay | Cloud (Google AI API) |

## Command Tiers

| Tier | Example | Latency | How It Works |
|------|---------|---------|-------------|
| **Fast Path** | "knight left", "fireball center" | ~170ms | On-device STT + parser + tap injection. No network. |
| **Queue Path** | "knight left then archers right" | ~170ms + wait | Buffers commands, auto-plays when card cycles into hand. |
| **Smart Path** | "defend", "follow up", "push right" | ~1.5s | Gemini Flash picks the optimal card using Opus playbook. |
| **Autopilot** | "autopilot" / "play for me" | ~4s/play | AI plays the entire match. Toggle on/off by saying "autopilot". |

## Quick Start

### Prerequisites

- Android phone (tested on Samsung Galaxy A35, Android 16)
- Clash Royale installed
- API keys: [Anthropic](https://console.anthropic.com/) (Claude) + [Google AI](https://aistudio.google.com/) (Gemini)

### 1. Download STT Model Assets

The speech recognition model (~180MB) is too large for git. Run the bootstrap script to download it:

**macOS / Linux:**
```bash
./scripts/bootstrap_sherpa_deps.sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_sherpa_deps.ps1
```

This downloads into gitignored paths:
- `app/src/main/assets/sherpa-onnx-zipformer-en-2023-04-01/` (STT model)
- `app/src/main/assets/silero_vad.onnx` (voice activity detection)
- `app/src/main/jniLibs/arm64-v8a/` (native libraries)

### 2. Configure API Keys

Copy the example and add your keys:
```bash
cp local.properties.example local.properties
```

Edit `local.properties`:
```properties
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
```

### 3. Build

```bash
./gradlew assembleDebug
```

### 4. Install

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### 5. Device Setup

1. **Allow restricted settings** (Android 13+): Settings > Apps > Clash Commander > menu (3 dots) > Allow restricted settings
2. Open Clash Commander and complete the setup checklist:
   - Grant overlay permission
   - Enable Accessibility Service (Settings > Accessibility > Clash Commander)
   - Grant microphone permission
   - Start screen capture
   - Start speech service (wait ~7s for model to load)
3. **Load your deck**: Tap "Share Deck from Clash Royale" — opens CR, go to your deck, tap Share, select Clash Commander
4. Wait for Opus strategic analysis (~20s)
5. Launch the overlay, tap "Listen", switch to Clash Royale
6. Start commanding!

## Architecture

```
Microphone -> Silero VAD -> Zipformer STT -> Command Parser -> Router
                                                                  |
                              +------------------+----------------+--------+
                              |                  |                |        |
                          FAST PATH         QUEUE PATH      SMART PATH  AUTOPILOT
                          (on-device)       (on-device)     (Gemini)    (Gemini)
                           ~170ms          ~170ms+wait       ~1.5s      ~4s/play
                              |                  |                |        |
                              +------------------+----------------+--------+
                                                  |
                                        AccessibilityService
                                          safeTap() / playCard()
                                                  |
                                           Clash Royale
```

**Pre-match:** Claude Opus 4.6 analyzes your deck and generates a strategic playbook (defense table, synergies, placement defaults). This playbook is injected into every Gemini Flash call as context.

**During match:** YOLOv8n-cls scans your hand at 5 FPS to identify which cards are available. The command router picks the fastest execution path for each voice command.

## Voice Command Examples

```
"knight left"              -> Fast Path: plays Knight at left bridge
"fireball center"          -> Fast Path: plays Fireball at center
"arrows left tower"        -> Fast Path: targets enemy left tower
"knight left then archers" -> Queue: plays Knight, queues Archers
"defend"                   -> Smart: AI picks best defensive card
"push right"               -> Smart: AI picks offensive card for right
"follow up"                -> Smart: supports last played card, same lane
"autopilot"                -> Toggle: AI plays entire match
"stop"                     -> Stops autopilot
```

## Tech Stack

- **Language:** Kotlin
- **UI:** Jetpack Compose (Material 3) with Clash Royale-themed design system
- **STT:** sherpa-onnx with Zipformer Transducer (int8) + Silero VAD + hotword biasing
- **Hand Detection:** YOLOv8n-cls (103 card classes, 98.1% val accuracy, ~3ms inference)
- **Strategy:** Claude Opus 4.6 (pre-match) + Gemini 3 Flash (real-time)
- **Input Injection:** Android AccessibilityService with touch-state gating
- **Screen Capture:** MediaProjection API

## Known Issues

- **Android 15/16 gesture freeze:** If you touch the screen while a voice command is executing, the gesture system may freeze. Hands-free operation is recommended — that's the whole point.
- **STT accuracy:** Some card names are challenging for the model. Giant and Arrows have lower recognition rates. The system includes 180+ aliases and 70+ compound aliases for STT misrecognitions.
- **No elixir tracking:** The system doesn't read your elixir bar. Cards may fail to play if you don't have enough elixir.

## Project Structure

```
app/src/main/java/com/yoyostudios/clashcompanion/
├── MainActivity.kt              # Compose UI, permissions, deck loading
├── accessibility/
│   └── ClashCompanionAccessibilityService.kt  # Tap injection (safeTap)
├── capture/
│   └── ScreenCaptureService.kt  # MediaProjection frames
├── speech/
│   └── SpeechService.kt        # sherpa-onnx STT pipeline
├── detection/
│   ├── HandDetector.kt          # Background hand scanning at 5 FPS
│   └── CardClassifier.kt       # YOLOv8n-cls inference
├── command/
│   ├── CommandRouter.kt         # Five-tier routing + Autopilot
│   └── CommandParser.kt        # Fuzzy matching + alias resolution
├── strategy/
│   └── OpusCoach.kt            # Claude Opus deck analysis
├── api/
│   ├── GeminiClient.kt          # Gemini 3 Flash API
│   ├── AnthropicClient.kt      # Claude API wrapper
│   └── RoboflowClient.kt      # Roboflow vision API
├── deck/
│   └── DeckManager.kt          # Deck share link parsing
├── overlay/
│   └── OverlayManager.kt       # Floating overlay HUD
├── ui/                          # Compose theme + components
└── util/
    ├── Coordinates.kt           # 1080x2340 zone map (40+ aliases)
    └── CardAliases.kt          # 180+ card aliases for STT
```

## License

MIT

## Disclaimer

Hackathon prototype built for the Supercell Global AI Game Hack. Respect Supercell's Terms of Service. Use at your own risk.
