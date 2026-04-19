---
youtube link: 
title: Mystical GM — Interactive Story Generator
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app_updated.py
pinned: false
suggested_hardware: zero-a10g
---

# Mystical GM — Interactive Story Generator

**CSCI 4052U — Machine Learning II | Final Project**

An end-to-end multimodal AI application where four neural networks collaborate to tell a branching, choose-your-own-adventure story. Each player turn produces narrative prose, and — when the scene calls for it — a freshly generated scene image and mood-adaptive background music.

---

## Demo

> **YouTube video:** https://youtu.be/MBf_KHYLxj4
> **Live Space:** https://huggingface.co/spaces/woolenelk/Mystic_GM

---

## Setup

### Option 1 — Hugging Face Spaces (recommended, no local GPU required)

1. Fork or duplicate this Space on Hugging Face.
2. Set hardware to **ZeroGPU (A10G)**.
3. The first cold boot downloads all four models (~14 GB total). Subsequent boots use the cache.
4. Navigate to the Space URL and start a story.

### Option 2 — Local GPU (≥16 GB VRAM)

Remove the `@spaces.GPU` decorator and `import spaces` from `app_updated.py`, then:

```bash
pip install -r requirements.txt
python app_updated.py
```

### Option 3 — Google Colab (T4, 16 GB VRAM)

Same edits as Option 2. The sequential CPU-offload path (each model moves to CUDA for its stage, then back to CPU) keeps peak VRAM under 16 GB.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

---

## The Problem

**Interactive multimodal storytelling:** given a short natural-language premise and a sequence of player decisions, produce an evolving narrative with coherent visuals and music — updating only when dramatically meaningful, not on every turn.

### Real-world motivation

Tabletop RPG Dungeon Masters spend significant time hand-crafting scene artwork and background music before a session. This preparation breaks down the moment players go off-script — entering an unplanned location or taking an unexpected turn means the DM has nothing prepared. Mystical GM solves this: when the story moves somewhere new, a matching scene image and mood-appropriate music are generated in seconds, with no preparation required.

### Why traditional approaches fail here

| Sub-problem | Traditional limit |
|---|---|
| Open-vocabulary narrative generation | Rule-based CYOA engines are hand-authored and cannot generalize to novel premises or actions outside their graph |
| Conditional cross-modal coordination | There is no rule-based way to decide *when* a scene change is dramatically significant enough to warrant new art |
| Affect classification from arbitrary images | Classical CV (HOG, color histograms) fails on stylistic or symbolic imagery outside its training distribution |
| Realistic text-conditioned music generation | Symbolic MIDI systems have no acoustic realism and no grounding in natural-language mood descriptions |

### The neural network approach

Each sub-problem is solved by a dedicated pretrained neural network operating in sequence:

1. **Qwen2.5-3B-Instruct** — reads the story history and player action, writes the next scene, and decides (via JSON flags) whether image and music should regenerate this turn.
2. **Stable Diffusion 1.5** — renders the scene as a 512×512 image when the LLM signals a visual change.
3. **CLIP ViT-L/14** — extracts the dominant mood from the generated image when the LLM does not supply an explicit hint.
4. **MusicGen-small** — generates ~28–30 seconds of mood-conditioned background music.

---

## Neural Network Components

### 1. Qwen2.5-3B-Instruct — Story Director

**Architecture.** Decoder-only causal language model with ~3 billion parameters. Key design choices: Rotary Position Embeddings (RoPE) for length generalization, Grouped Query Attention (GQA) to reduce KV-cache memory, SwiGLU feed-forward activations, and RMSNorm. 32K token context window.

**Training.** Pretrained on ~18 trillion tokens of multilingual text (web, books, code, scientific literature). Post-trained with supervised fine-tuning and RLHF for instruction following, including structured JSON output tasks.

**Use in this project.** The LLM is prompted with a detailed system prompt and few-shot examples enforcing a strict JSON schema. Each turn it outputs: scene prose (2–3 sentences, second person), three distinct player choices, `visual_change` and `audio_change` booleans, an SD-optimized image prompt, an optional mood hint, and a `style_anchor` string that is prepended to all future image prompts to maintain visual consistency. Two-attempt parse-retry logic with a hardcoded fallback guards against malformed output.

**Citation.** Yang et al., *"Qwen2.5 Technical Report,"* 2024. arXiv:2412.15115.

---

### 2. Stable Diffusion 1.5 — Scene Image Generation

**Architecture.** Latent diffusion model. A variational autoencoder (VAE) encodes 512×512 RGB images into 64×64×4 latent tensors; a U-Net denoiser iteratively removes noise from those latents, conditioned on text via cross-attention layers; a frozen CLIP text encoder produces the conditioning embeddings. Inference: 30 DDIM steps with classifier-free guidance (scale 7.5) starting from Gaussian noise, decoded back to pixels by the VAE.

**Training.** Pretrained on LAION-5B (~5 billion image-text pairs), fine-tuned on LAION-Aesthetics v2.5+ (~600M high-quality pairs). Weights loaded in float16.

**Use in this project.** Only fires when `visual_change=true`. The image prompt is constructed by prepending the `style_anchor` established on turn 1, ensuring consistent art direction across the whole story regardless of scene content.

**Citation.** Rombach et al., *"High-Resolution Image Synthesis with Latent Diffusion Models,"* CVPR 2022. Schuhmann et al., *"LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models,"* NeurIPS 2022.

---

### 3. CLIP ViT-L/14 — Mood Extraction

**Architecture.** Contrastive dual-encoder. Image encoder: Vision Transformer (ViT) with 14×14 patch embeddings over a 224×224 input, 24 Transformer layers, 1024 hidden dim, projected to a 768-dim shared embedding space. Text encoder: 12-layer Transformer projected to the same 768-dim space. Trained end-to-end with symmetric InfoNCE contrastive loss to maximize cosine similarity between matching image-text pairs.

**Training.** WIT (WebImageText), ~400 million curated image-text pairs from the web.

**Use in this project.** Zero-shot mood classification. Ten mood phrases (e.g. *"dark and ominous"*, *"mystical and ethereal"*) are encoded once into a `[10, 768]` embedding matrix and cached. Each generated scene image is encoded to a `[1, 768]` vector; cosine similarity against the mood matrix yields the top-3 mood labels that drive MusicGen.

**Citation.** Radford et al., *"Learning Transferable Visual Models From Natural Language Supervision,"* ICML 2021.

---

### 4. MusicGen-small — Music Generation

**Architecture.** Single-stage autoregressive Transformer (~300M parameters) over discrete audio tokens. Audio is tokenized by a pretrained EnCodec codec: 4 residual vector-quantization codebooks at 50 Hz frame rate, 32 kHz sample rate. A T5 text encoder provides cross-attention conditioning from the mood prompt. Delay-pattern interleaving allows all four codebook streams to be generated in one pass.

**Training.** ~20,000 hours of licensed music with text descriptions.

**Use in this project.** Only fires when `audio_change=true` (or on the very first turn). `max_new_tokens=600` yields ~28–30 seconds of audio at 50 Hz. The mood prompt is constructed from the top-3 CLIP-extracted moods (or the LLM's hint when provided) using a fixed template: *"cinematic background music, {primary} mood, {secondary} atmosphere, {tertiary} feeling, orchestral, film score."*

**Citation.** Copet et al., *"Simple and Controllable Music Generation,"* NeurIPS 2023.

---

## End-to-End Application Pipeline

### Software Architecture

```
User (Gradio UI)
    │
    ▼
take_turn()  ←─ @spaces.GPU generator, yields 7-tuples for progressive UI updates
    │
    ├─[Always]─────────────────── Stage 1: Qwen2.5-3B-Instruct
    │                              history + action → JSON (scene, flags, prompts)
    │
    ├─[visual_change=true]──────── Stage 2: Stable Diffusion 1.5
    │                              style_anchor + image_prompt → PIL Image
    │
    └─[audio_change=true]──────── Stage 3a: CLIP ViT-L/14
                                   PIL Image → top-3 mood labels
                                       │
                                   Stage 3b: MusicGen-small
                                   mood labels → WAV filepath
```

The LLM acts as the **director**: its boolean output flags (`visual_change`, `audio_change`) gate the expensive downstream models, saving ~35–45 seconds of generation per turn when the scene has not meaningfully changed.

### Tensor Encoding per Stage

| Stage | Input | Encoding | Tensor shape | Output |
|---|---|---|---|---|
| Qwen2.5 | History + action (str) | Tokenizer → causal decode | `[1, seq_len]` → `[1, ≤512]` new tokens | JSON string |
| Stable Diffusion | Image prompt (str) | SD CLIP tokenizer → 30-step latent denoising → VAE decode | `[1, 4, 64, 64]` latents | PIL image |
| CLIP | PIL image | `clip_preprocess` → ViT encode | `[1, 3, 224, 224]` → `[1, 768]` | Cosine sim vs `[10, 768]` |
| MusicGen | Mood labels (list[str]) | T5 encode → autoregressive decode | `[1, 1, ~960 000]` float waveform | WAV file |

### Application Code Interfaces

- **CPU-first model loading.** All four models are instantiated at import time on CPU. ZeroGPU's `@spaces.GPU` decorator moves each model to CUDA for its stage only, then returns it to CPU, keeping peak VRAM usage under 16 GB.
- **Cached mood embeddings.** The `[10, 768]` CLIP text embedding matrix is computed once on the first GPU call and stored in a module-level tensor (`_mood_text_embeddings`), avoiding repeated encoding across turns.
- **Progressive UI updates.** `take_turn` is a Python generator yielding a 7-tuple after each stage so Gradio can stream partial results to the user (scene text appears before image, image before music).
- **Conditional generation.** SD only runs if `visual_change=true`; CLIP+MusicGen only run if `audio_change=true` or it is turn 1 with no audio yet.
- **OOM safety.** Each stage is wrapped in `try/except torch.cuda.OutOfMemoryError` with `torch.cuda.empty_cache()` and `gc.collect()`. A failing stage is skipped gracefully; the prior image/audio are retained.
- **Temp file management.** WAV files are written to `tempfile.NamedTemporaryFile(suffix=".wav", delete=False)`. The previous WAV is deleted before the new one is stored; `cleanup_audio()` is also wired to `scene_audio.clear`.

---

## Models Summary

| Model | Role | Checkpoint | Approx. size |
|---|---|---|---|
| Qwen2.5-3B-Instruct | Story direction & JSON control | `Qwen/Qwen2.5-3B-Instruct` | ~6 GB |
| Stable Diffusion 1.5 | Scene image generation | `runwayml/stable-diffusion-v1-5` | ~4 GB |
| CLIP ViT-L/14 | Image → mood classification | `ViT-L-14` via `open_clip` | ~1.7 GB |
| MusicGen-small | Mood → background music | `facebook/musicgen-small` | ~2.2 GB |

---

## Repository Structure

```
.
├── app_updated.py      # Main application (all four models + Gradio UI)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Limitations

- Fixed 10-label mood vocabulary; out-of-vocabulary moods collapse onto the nearest label.
- Music clips (~28–30 s) loop but have no temporal alignment to story beats within a scene.
- English-only prompts (inherited from all four models' training distributions).
- No fine-tuning; stylistic control is limited to what the base models already know.
- Story state is session-only — lost on page refresh (`gr.State`).
