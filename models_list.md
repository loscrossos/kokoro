# Model Checkpoints & Assets Inventory

This document provides a comprehensive inventory of all model weights, checkpoints, configuration files, vocoders, tokenizers, and voice packs used across the Kokoro repository.

Kokoro uses a **unified, offline-friendly checkpoint directory architecture**. All checkpoints default to the root `checkpoints/` directory, organized under a subfolder matching the repository name of their source:
```
checkpoints/<reponame>/<subpath>/<filename>
```

---

## 1. Primary Model: `hexgrad/Kokoro-82M`

- **Source Repository:** `https://huggingface.co/hexgrad/Kokoro-82M`
- **Subfolder Destination:** `checkpoints/Kokoro-82M/`

### 1.1 Architecture & Weights

| Asset Name | Source URL | Expected Local Path | Fallback Path | Description |
|------------|------------|---------------------|---------------|-------------|
| `kokoro-v1_0.pth` | `https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth` | `checkpoints/Kokoro-82M/kokoro-v1_0.pth` | `checkpoints/kokoro-v1_0.pth` | Kokoro-82M v1.0 PyTorch model checkpoint (~327 MB) |
| `config.json` | `https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json` | `checkpoints/Kokoro-82M/config.json` | `checkpoints/config.json` | Model configuration (vocab, hidden dimensions, style dim, iSTFTNet parameters) |

### 1.2 Voice Packs (`voices/*.pt`)

All voices are 510-frame style tensor vectors stored in PyTorch format (`.pt`).

- **Base URL:** `https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{voice_name}.pt`
- **Expected Local Path:** `checkpoints/Kokoro-82M/voices/{voice_name}.pt`
- **Fallback Path:** `checkpoints/voices/{voice_name}.pt`

#### American English (`a`, `en-us`)
- Female:
  - `af_heart.pt` (Default voice)
  - `af_alloy.pt`
  - `af_aoede.pt`
  - `af_bella.pt`
  - `af_jessica.pt`
  - `af_kore.pt`
  - `af_nicole.pt`
  - `af_nova.pt`
  - `af_river.pt`
  - `af_sarah.pt`
  - `af_sky.pt`
- Male:
  - `am_adam.pt`
  - `am_echo.pt`
  - `am_eric.pt`
  - `am_fenrir.pt`
  - `am_liam.pt`
  - `am_michael.pt`
  - `am_onyx.pt`
  - `am_puck.pt`
  - `am_santa.pt`

#### British English (`b`, `en-gb`)
- Female:
  - `bf_alice.pt`
  - `bf_emma.pt`
  - `bf_isabella.pt`
  - `bf_lily.pt`
- Male:
  - `bm_daniel.pt`
  - `bm_fable.pt`
  - `bm_george.pt`
  - `bm_lewis.pt`

#### Spanish (`e`, `es`)
- Female: `ef_dora.pt`
- Male: `em_alex.pt`, `em_santa.pt`

#### French (`f`, `fr-fr`)
- Female: `ff_siwis.pt`

#### Hindi (`h`, `hi`)
- Female: `hf_alpha.pt`, `hf_beta.pt`
- Male: `hm_omega.pt`, `hm_psi.pt`

#### Italian (`i`, `it`)
- Female: `if_sara.pt`
- Male: `im_nicola.pt`

#### Japanese (`j`, `ja`)
- Female: `jf_alpha.pt`, `jf_gongitsune.pt`, `jf_nezumi.pt`, `jf_tebukuro.pt`
- Male: `jm_kumo.pt`

#### Brazilian Portuguese (`p`, `pt-br`)
- Female: `pf_dora.pt`
- Male: `pm_alex.pt`, `pm_santa.pt`

#### Mandarin Chinese (`z`, `zh`)
- Female: `zf_xiaobei.pt`, `zf_xiaoni.pt`, `zf_xiaoxiao.pt`, `zf_xiaoyi.pt`
- Male: `zm_yunjian.pt`, `zm_yunxi.pt`, `zm_yunxia.pt`, `zm_yunyang.pt`

---

## 2. Multilingual Model: `hexgrad/Kokoro-82M-v1.1-zh`

- **Source Repository:** `https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh`
- **Subfolder Destination:** `checkpoints/Kokoro-82M-v1.1-zh/`

| Asset Name | Source URL | Expected Local Path | Fallback Path | Description |
|------------|------------|---------------------|---------------|-------------|
| `kokoro-v1_1-zh.pth` | `https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/kokoro-v1_1-zh.pth` | `checkpoints/Kokoro-82M-v1.1-zh/kokoro-v1_1-zh.pth` | `checkpoints/kokoro-v1_1-zh.pth` | Kokoro-82M v1.1 Mandarin Chinese model weights |
| `config.json` | `https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/config.json` | `checkpoints/Kokoro-82M-v1.1-zh/config.json` | `checkpoints/config.json` | Mandarin v1.1 configuration |
| Chinese Voice Packs | `https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/voices/{voice}.pt` | `checkpoints/Kokoro-82M-v1.1-zh/voices/{voice}.pt` | `checkpoints/voices/{voice}.pt` | Chinese style vectors (`zf_*`, `zm_*`) |

---

## 3. Community Models & Language Extensions: `cryptomilk/kokoro-german-kerstin`

- **Source Repository:** `https://huggingface.co/cryptomilk/kokoro-german-kerstin`
- **Subfolder Destination:** `checkpoints/kokoro-german-kerstin/`
- **Language Routing:** German (`de`, `d`) via `espeak.EspeakG2P(language='de')`

| Asset Name | Source URL | Expected Local Path | Fallback Path | Description |
|------------|------------|---------------------|---------------|-------------|
| German Voice Packs (`df_kerstin.pt`, etc.) | `https://huggingface.co/cryptomilk/kokoro-german-kerstin/resolve/main/voices/{voice}.pt` | `checkpoints/kokoro-german-kerstin/voices/{voice}.pt` | `checkpoints/voices/{voice}.pt` | Community German voice tensor |

---

## 4. Sub-components, Vocoders & Tokenizers

| Component | Upstream / Package | Implementation | Configuration & Model Weights |
|-----------|--------------------|----------------|-------------------------------|
| **PLBERT Text Encoder** | `transformers` | `CustomAlbert` (`kokoro/modules.py`) | Configured via `config.json["plbert"]` and vocabulary in `config.json["vocab"]` |
| **Prosody Predictor** | StyleTextToSpeech2 | `ProsodyPredictor` (`kokoro/modules.py`) | Predictor layers loaded from `kokoro-v1_0.pth` |
| **Vocoder / Decoder** | iSTFTNet | `Decoder`, `TorchSTFT`, `CustomSTFT` (`kokoro/istftnet.py`, `kokoro/custom_stft.py`) | 24 kHz neural vocoder decoder weights inside `kokoro-v1_0.pth` |
| **G2P / Phonemizer (EN)** | `misaki[en]` | `en.G2P` with `espeak.EspeakFallback` | English phoneme converter and dictionary |
| **G2P / Phonemizer (JA)** | `misaki[ja]` | `ja.JAG2P` | Japanese phonemizer |
| **G2P / Phonemizer (ZH)** | `misaki[zh]` | `zh.ZHG2P` | Mandarin phonemizer |
| **G2P / Fallback** | `espeak-ng` | `espeak.EspeakG2P` | Multi-language fallback phonemizer |

---

## 4. Exported & Web Runtime Formats

### 4.1 ONNX & Triton
- **Export Script:** `examples/export.py`, `examples/make_triton_compatible.py`
- **Output Artifacts:**
  - `onnx/kokoro.onnx` or `checkpoints/onnx/kokoro.onnx`
  - `onnx/kokoro_triton.onnx` or `checkpoints/onnx/kokoro_triton.onnx`

### 4.2 Web Runtime (`kokoro.js`)
- **Source Repository:** `onnx-community/Kokoro-82M-ONNX`
- **Local Assets:** `kokoro.js/voices/*.bin` (54 binary voice vectors for browser/WebAssembly inference)

---

## 5. Offline Pre-Caching Instructions

To prepare the repository for completely offline operation, download the desired model files and place them into the `checkpoints/` directory following the source repository subfolder structure:

```bash
# Example structure:
checkpoints/
├── Kokoro-82M/
│   ├── config.json
│   ├── kokoro-v1_0.pth
│   └── voices/
│       ├── af_heart.pt
│       ├── af_bella.pt
│       └── ...
└── Kokoro-82M-v1.1-zh/
    ├── config.json
    ├── kokoro-v1_1-zh.pth
    └── voices/
        └── ...
```

When these files exist, `kokoro` resolves and loads them immediately without any external network calls.
