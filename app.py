import gc
import json
import re
import spaces
import torch
import numpy as np
import scipy.io.wavfile
import tempfile

from diffusers import StableDiffusionPipeline
import open_clip
from transformers import (
    MusicgenForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from PIL import Image
import gradio as gr

# ---------------------------------------------------------------------------
# Model loading (CPU at module level — ZeroGPU moves to GPU inside @spaces.GPU)
# ---------------------------------------------------------------------------

sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai"
)
clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

music_model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small"
)
music_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

llm_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    trust_remote_code=True,
)
llm_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# ---------------------------------------------------------------------------
# Mood vocabulary
# ---------------------------------------------------------------------------

MOOD_LABELS = [
    "dark and ominous",
    "peaceful and serene",
    "epic and heroic",
    "melancholic and somber",
    "whimsical and playful",
    "tense and suspenseful",
    "warm and romantic",
    "cold and desolate",
    "chaotic and intense",
    "mystical and ethereal",
]

# Cached mood text embeddings — computed on first GPU call, reused thereafter
_mood_text_embeddings: torch.Tensor | None = None


def _get_mood_embeddings() -> torch.Tensor:
    """Lazily compute and cache CLIP text embeddings for MOOD_LABELS.

    Must be called inside an @spaces.GPU context so 'cuda' is available.

    Returns:
        torch.Tensor: Normalized mood embeddings [10, 768] on CUDA.
    """
    global _mood_text_embeddings
    if _mood_text_embeddings is None:
        mood_tokens = clip_tokenizer(MOOD_LABELS).to("cuda")
        with torch.no_grad():
            emb = clip_model.encode_text(mood_tokens)          # [10, 768]
            _mood_text_embeddings = emb / emb.norm(dim=-1, keepdim=True)
    return _mood_text_embeddings


# ---------------------------------------------------------------------------
# LLM: system prompt, fallback, and JSON parsing
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = """\
You are a storyteller for an interactive choose-your-own-adventure game. \
Each turn you receive the story so far and the player's action. \
Respond with a single JSON object — no other text, no markdown, no code fences.

JSON schema (all fields required):
{
  "scene": "<2-3 vivid sentences of narrative prose in second person>",
  "choices": ["<action 1>", "<action 2>", "<action 3>"],
  "visual_change": <true | false>,
  "audio_change": <true | false>,
  "image_prompt": "<SD-optimized concrete visual noun phrases, or empty string if visual_change=false>",
  "audio_mood_hint": "<mood string or null>",
  "style_anchor": "<visual style descriptor, keep consistent across turns>"
}

SCENE TEXT rules:
- 2-3 sentences maximum. Second person ("You see...", "The air carries...").
- Focus on visual atmosphere: lighting, environment, weather, textures, sensory detail.

IMAGE PROMPT rules:
- Concrete noun phrases only: subject + setting + lighting + composition.
- Good: "crumbling lighthouse at dusk, warm orange rim light, wide low angle, oil-painted texture"
- Bad: "A scene where the hero approaches the lighthouse in the evening"
- Leave empty string if visual_change=false.

VISUAL_CHANGE rules:
- true: player enters a new location, dramatic transformation occurs, time of day shifts, \
major scenery change.
- false: player examines an object, has a conversation, waits, thinks. Setting is unchanged.

AUDIO_CHANGE rules:
- true: emotional mood meaningfully shifts, OR visual_change is true.
- false: current mood is continuous with the previous scene.

AUDIO_MOOD_HINT rules:
- Optional string or null. If you have a clear mood in mind, name it exactly from this list:
  "dark and ominous", "peaceful and serene", "epic and heroic", "melancholic and somber",
  "whimsical and playful", "tense and suspenseful", "warm and romantic", "cold and desolate",
  "chaotic and intense", "mystical and ethereal"
- Set null to let the app extract mood from the generated image automatically.

STYLE_ANCHOR rules:
- A short visual style descriptor (e.g. "dark gothic oil painting, muted earth tones, candlelight").
- Establish it on turn 1 from the genre/premise. Keep it roughly consistent across all turns.

CHOICES rules:
- 3 meaningfully distinct options — different goals, strategies, or directions.
- Not minor variations of the same action. General audience only.

---
EXAMPLES

Example 1 — Opening turn, fantasy, visual_change=true:
User: Genre: fantasy
Action: I step through the ancient doorway into the forgotten temple.
{"scene": "Columns of pale moonstone rise on either side, carved with spiraling runes that pulse with faint silver light. The air is cold and carries the smell of old stone and something floral — like flowers that bloomed centuries ago. Ahead, a central altar glows with a steady blue flame that casts no shadow.", "choices": ["Approach the altar and examine the flame", "Study the runes on the nearest column", "Listen carefully for sounds deeper in the temple"], "visual_change": true, "audio_change": true, "image_prompt": "ancient moonstone temple interior, glowing blue altar flame, spiral rune columns, dramatic chiaroscuro lighting, wide shot", "audio_mood_hint": "mystical and ethereal", "style_anchor": "luminous fantasy oil painting, cool silver-blue palette with warm moonlight, soft volumetric lighting"}

Example 2 — Mid-story, conversation, visual_change=false:
User: Action: I ask the innkeeper about the missing merchant.
{"scene": "The innkeeper sets down her cloth and glances toward the door before leaning in. 'He came through three nights ago,' she murmurs, 'paid for a week upfront, and was gone by morning — room untouched.' Her eyes say she knows more than she is saying.", "choices": ["Press her for more details", "Ask if anyone else noticed the merchant", "Thank her and search the merchant's room"], "visual_change": false, "audio_change": false, "image_prompt": "", "audio_mood_hint": null, "style_anchor": "dark mystery oil painting, warm amber tavern lamplight, muted earth tones"}

Example 3 — Dramatic event, mood shift, visual_change=true:
User: Action: I pull the lever on the wall.
{"scene": "A grinding roar fills the chamber as the floor trembles beneath your feet. Stone blocks crash into the river below, revealing a hidden passage in the far wall — but the ceiling has begun to descend, inch by grinding inch.", "choices": ["Sprint for the hidden passage immediately", "Search for a second lever to stop the ceiling", "Grab your pack and brace against the wall"], "visual_change": true, "audio_change": true, "image_prompt": "collapsing stone chamber with descending ceiling, hidden passage in far wall, torchlight and dust clouds, dramatic low angle", "audio_mood_hint": "chaotic and intense", "style_anchor": "dark fantasy oil painting, cool silver and blue palette, chiaroscuro lighting"}
"""

LLM_FALLBACK = {
    "scene": "You pause, taking stock of your surroundings. The path ahead is unclear, but you sense the story is not over yet.",
    "choices": ["Look around carefully", "Move forward", "Wait and listen"],
    "visual_change": False,
    "audio_change": False,
    "image_prompt": "",
    "audio_mood_hint": None,
    "style_anchor": "painterly realism, natural earth tones, soft diffuse light",
}

# Neutral fallback mood used only when no image exists and the LLM provided no hint
_MOOD_FALLBACK = ["peaceful and serene"]


def _parse_llm_response(text: str) -> dict:
    """Extract and parse the first JSON object found in an LLM response string."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response")
    return json.loads(match.group())


def call_llm(history: list[dict], action: str, genre: str) -> dict:
    """Call Qwen2.5-3B-Instruct and return a validated story JSON dict.

    Args:
        history: List of {"scene": str, "action": str} dicts from prior turns.
        action:  The player's current action string.
        genre:   Story genre (e.g. "fantasy").

    Returns:
        dict matching the LLM JSON schema, or LLM_FALLBACK on two consecutive failures.
    """
    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": "assistant", "content": turn["scene"]})
        messages.append({"role": "user", "content": turn["action"]})
    if action.strip():
        user_content = f"Genre: {genre}\nAction: {action}"
    else:
        user_content = (
            f"Genre: {genre}\n"
            "No premise was provided. Invent an original opening scene from scratch — "
            "you have complete creative freedom over the setting, character, and situation."
        )
    messages.append({"role": "user", "content": user_content})

    def _generate(msgs: list[dict], temperature: float) -> str:
        text = llm_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = llm_tokenizer([text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_ids = llm_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True,
                pad_token_id=llm_tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return llm_tokenizer.decode(new_ids, skip_special_tokens=True)

    # First attempt
    response = _generate(messages, temperature=0.8)
    try:
        return _parse_llm_response(response)
    except (json.JSONDecodeError, ValueError):
        pass

    # Retry with correction prompt at lower temperature
    messages.append({"role": "assistant", "content": response})
    messages.append({
        "role": "user",
        "content": (
            "Your previous response was not valid JSON. "
            "Respond with only the JSON object and nothing else."
        ),
    })
    response2 = _generate(messages, temperature=0.3)
    try:
        return _parse_llm_response(response2)
    except (json.JSONDecodeError, ValueError):
        return LLM_FALLBACK.copy()


# ---------------------------------------------------------------------------
# Stage 1: Image generation
# ---------------------------------------------------------------------------

def generate_image(prompt: str) -> Image.Image:
    """Generate a scene image from a text prompt using Stable Diffusion 1.5.

    Args:
        prompt: User-supplied text description of the scene.

    Returns:
        PIL.Image.Image: Generated RGB image (default 512x512).
    """
    result = sd_pipe(prompt, num_inference_steps=30, guidance_scale=7.5)
    return result.images[0]


# ---------------------------------------------------------------------------
# Stage 2: Mood extraction via CLIP
# ---------------------------------------------------------------------------

def extract_mood(image: Image.Image) -> tuple[list[str], str]:
    """Extract the top-3 mood labels from an image using CLIP cosine similarity.

    Pipeline:
        PIL Image -> clip_preprocess -> tensor [1, 3, 224, 224] (CLIP normalized)
        -> clip_model.encode_image -> image embedding [1, 768]
        -> cosine similarity against pre-encoded mood_text_embeddings [10, 768]
        -> top-3 indices -> mood label strings

    Args:
        image: PIL.Image.Image — the generated scene image.

    Returns:
        tuple:
            - list[str]: Top-3 mood label strings.
            - str: Human-readable display text listing the moods.
    """
    image_input = clip_preprocess(image).unsqueeze(0).to("cuda")  # [1, 3, 224, 224]

    with torch.no_grad():
        image_embedding = clip_model.encode_image(image_input)         # [1, 768]
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

    mood_embeddings = _get_mood_embeddings()                            # [10, 768]
    similarities = (image_embedding @ mood_embeddings.T).squeeze(0)    # [10]
    top3_indices = similarities.topk(3).indices.tolist()
    top3_moods = [MOOD_LABELS[i] for i in top3_indices]

    display_text = "Detected moods:\n" + "\n".join(f"  - {m}" for m in top3_moods)
    return top3_moods, display_text


# ---------------------------------------------------------------------------
# Stage 3: Music generation via MusicGen
# ---------------------------------------------------------------------------

def generate_music(mood_labels: list[str]) -> str:
    """Generate a 10-15 second music clip conditioned on mood labels.

    The mood labels are formatted into a descriptive music prompt before being
    fed to MusicGen-small, ensuring the audio is conditioned on the *image*
    content (via CLIP), not on the raw user text prompt.

    Args:
        mood_labels: list[str] — top-3 mood strings from extract_mood().

    Returns:
        str: Filepath to the generated .wav file (32000 Hz, mono int16).
    """
    primary, secondary, tertiary = (mood_labels + [""] * 3)[:3]
    music_prompt = (
        f"cinematic background music, {primary} mood, "
        f"{secondary} atmosphere, {tertiary} feeling, orchestral, film score"
    )

    inputs = music_processor(
        text=[music_prompt],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # max_new_tokens=600 yields ~28-30 seconds at MusicGen's 50 Hz frame rate
    with torch.no_grad():
        audio_values = music_model.generate(**inputs, max_new_tokens=600)  # ~28-30 s at 50 Hz

    audio_np = audio_values[0, 0].cpu().numpy()             # [samples], float32
    audio_np = np.clip(audio_np, -1.0, 1.0)                # guard against out-of-range values
    audio_np = (audio_np * 32767).astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    scipy.io.wavfile.write(tmp.name, rate=32000, data=audio_np)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Orchestrator (ZeroGPU entry point)
# ---------------------------------------------------------------------------

@spaces.GPU
def take_turn(action: str, state: dict, chatbot: list):
    """Generator: one story turn with progressive UI updates.

    Yields 7-tuples:
        (chatbot, state, btn0_update, btn1_update, btn2_update, image_update, audio_update)

    Stage 1 (LLM):   yields scene text + disabled choice buttons immediately.
    Stage 2 (SD):    yields new image only if result["visual_change"] is True.
    Stage 3 (CLIP+MusicGen): yields new audio only if result["audio_change"] is True.
    Final yield:     re-enables and labels choice buttons.
    """
    _no_change = gr.update()
    if not (action or "").strip() and state["turn"] > 0:
        return
    # Disable buttons during generation
    yield (
        chatbot, state,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        _no_change, _no_change,
    )

    # ── Stage 1: LLM ────────────────────────────────────────────────────────
    result = LLM_FALLBACK.copy()
    try:
        llm_model.to("cuda")
        result = call_llm(state["history"], action, state["genre"])
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
    finally:
        llm_model.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

    if result.get("style_anchor") and not state["style_anchor"]:
        state["style_anchor"] = result["style_anchor"]

    chatbot = chatbot + [
        {"role": "user", "content": f"> {action}"},
        {"role": "assistant", "content": result["scene"]},
    ]
    state["history"].append({"scene": result["scene"], "action": action})
    state["current_choices"] = result.get("choices", LLM_FALLBACK["choices"])
    state["turn"] += 1

    yield (
        chatbot, state,
        gr.update(value=state["current_choices"][0], interactive=False),
        gr.update(value=state["current_choices"][1], interactive=False),
        gr.update(value=state["current_choices"][2], interactive=False),
        _no_change, _no_change,
    )

    # ── Stage 2: Stable Diffusion (conditional; always on opener) ────────────
    is_opener = state["turn"] == 1  # state["turn"] was just incremented
    if result.get("visual_change", False) or is_opener:
        image_prompt = (
            result.get("image_prompt")
            or state.get("style_anchor")
            or "atmospheric establishing shot, cinematic composition"
        )
        full_prompt = (
            f"{state['style_anchor']}, {image_prompt}"
            if state["style_anchor"] and state["style_anchor"] not in image_prompt
            else image_prompt
        )
        new_image = state.get("last_image")
        try:
            sd_pipe.to("cuda")
            new_image = generate_image(full_prompt)
            state["last_image"] = new_image
            state["last_image_prompt"] = full_prompt
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
        finally:
            sd_pipe.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()
    
        yield (
            chatbot, state,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value=new_image),
            _no_change,
        )

    # ── Stage 3: CLIP + MusicGen (conditional) ───────────────────────────────
    audio_change = result.get("audio_change", False)
    first_turn_no_audio = state["turn"] == 1 and state.get("last_audio") is None

    if audio_change or first_turn_no_audio:
        hint = result.get("audio_mood_hint")
        current_image = state.get("last_image")

        if hint:
            mood_labels = [hint]
        elif current_image is not None:
            mood_labels = ["peaceful and serene"]  # safe default before CLIP runs
            try:
                clip_model.to("cuda")
                mood_labels, _ = extract_mood(current_image)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
            finally:
                clip_model.to("cpu")
                torch.cuda.empty_cache()
                gc.collect()
        else:
            mood_labels = _MOOD_FALLBACK

        new_audio = state.get("last_audio")
        try:
            music_model.to("cuda")
            new_audio = generate_music(mood_labels)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
        finally:
            music_model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

        # Clean up previous temp file
        old_audio = state.get("last_audio")
        if old_audio and old_audio != new_audio:
            cleanup_audio(old_audio)
        state["last_audio"] = new_audio

        yield (
            chatbot, state,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            _no_change,
            gr.update(value=new_audio),
        )

    # ── Final: re-enable choice buttons ─────────────────────────────────────
    choices = state["current_choices"]
    yield (
        chatbot, state,
        gr.update(value=choices[0], interactive=True, visible=True),
        gr.update(value=choices[1], interactive=True, visible=True),
        gr.update(value=choices[2], interactive=True, visible=True),
        _no_change, _no_change,
    )


# ---------------------------------------------------------------------------
# Temp file cleanup on audio component clear
# ---------------------------------------------------------------------------

def cleanup_audio(audio_path: str | None) -> None:
    """Remove a previously generated wav temp file when the user clears audio."""
    import os
    if audio_path and os.path.isfile(audio_path):
        try:
            os.remove(audio_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _fresh_state(genre: str) -> dict:
    return {
        "history": [],
        "current_choices": ["", "", ""],
        "turn": 0,
        "genre": genre,
        "style_anchor": "",
        "last_image": None,
        "last_image_prompt": "",
        "last_audio": None,
    }


def reset_state(genre: str, premise: str):
    """Sync reset called before the first take_turn on a new story."""
    return (
        _fresh_state(genre),
        [],                                                    # chatbot
        gr.update(value=None),                                 # scene_image
        gr.update(value=None),                                 # scene_audio
        gr.update(value="—", interactive=False, visible=False),
        gr.update(value="—", interactive=False, visible=False),
        gr.update(value="—", interactive=False, visible=False),
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Interactive Story") as demo:
    gr.Markdown(
        "# Interactive Story Generator\n"
        "Choose your starting genre, describe your opening scene, and let four neural networks "
        "tell the story with you — image and music update only when the scene calls for it."
    )

    # ── Top: genre + premise + start ────────────────────────────────────────
    with gr.Row():
        genre_dd = gr.Dropdown(
            choices=["fantasy", "sci-fi", "horror", "mystery", "slice-of-life", "misc"],
            value="fantasy",
            label="Starting Genre",
            scale=1,
        )
        premise_box = gr.Textbox(
            label="How does your story begin?",
            placeholder="A traveler arrives at a fog-shrouded village with no memory of how they got there…",
            lines=2,
            scale=3,
        )
        start_btn = gr.Button("Start New Story", variant="primary", scale=1)

    # ── Middle: transcript + current media ──────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(type="messages", height=600, label="Story")
        with gr.Column(scale=1):
            scene_image = gr.Image(label="Current Scene", type="pil", interactive=False)
            scene_audio = gr.Audio(
                label="Scene Music",
                type="filepath",
                autoplay=True,
                loop=True,
                interactive=False,
            )

    # ── Bottom: choice buttons + free text ──────────────────────────────────
    with gr.Row():
        choice_btn_0 = gr.Button("—", interactive=False, visible=False, variant="secondary")
        choice_btn_1 = gr.Button("—", interactive=False, visible=False, variant="secondary")
        choice_btn_2 = gr.Button("—", interactive=False, visible=False, variant="secondary")

    with gr.Row():
        free_text = gr.Textbox(
            label="Or do something else…",
            placeholder="Describe your own action",
            scale=4,
            lines=1,
        )
        free_submit = gr.Button("Submit", scale=1)

    # Shared state
    state = gr.State(_fresh_state("fantasy"))

    # Shared output list (order must match take_turn yields)
    _turn_outputs = [chatbot, state, choice_btn_0, choice_btn_1, choice_btn_2, scene_image, scene_audio]

    # ── Start New Story ──────────────────────────────────────────────────────
    start_btn.click(
        fn=reset_state,
        inputs=[genre_dd, premise_box],
        outputs=[state, chatbot, scene_image, scene_audio, choice_btn_0, choice_btn_1, choice_btn_2],
    ).then(
        fn=take_turn,
        inputs=[premise_box, state, chatbot],
        outputs=_turn_outputs,
    )

    # ── Choice buttons ───────────────────────────────────────────────────────
    choice_btn_0.click(
        fn=take_turn,
        inputs=[choice_btn_0, state, chatbot],
        outputs=_turn_outputs,
    )
    choice_btn_1.click(
        fn=take_turn,
        inputs=[choice_btn_1, state, chatbot],
        outputs=_turn_outputs,
    )
    choice_btn_2.click(
        fn=take_turn,
        inputs=[choice_btn_2, state, chatbot],
        outputs=_turn_outputs,
    )

    # ── Free-text action ─────────────────────────────────────────────────────
    free_submit.click(
        fn=take_turn,
        inputs=[free_text, state, chatbot],
        outputs=_turn_outputs,
    ).then(fn=lambda: "", outputs=free_text)

    free_text.submit(
        fn=take_turn,
        inputs=[free_text, state, chatbot],
        outputs=_turn_outputs,
    ).then(fn=lambda: "", outputs=free_text)

    # ── Audio cleanup ────────────────────────────────────────────────────────
    scene_audio.clear(fn=cleanup_audio, inputs=scene_audio)


if __name__ == "__main__":
    demo.launch()