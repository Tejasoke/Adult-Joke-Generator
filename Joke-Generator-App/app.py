
import streamlit as st
import torch
import re
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="AI Joke Generator",
    page_icon="😂"
)

st.title("😂 AI Joke Generator")
st.caption("Give me a topic, I'll try to be funny.")
st.markdown("---")
st.caption("Made  by TEJAS OKE & KARTIKEY SINGH")

MODEL_ID = "tejasoke/joke-generator-gpt2"

# ----------------------------------
# Content Filter
# ----------------------------------
NSFW_KEYWORDS = [
    "sex", "porn", "xxx", "nude", "naked",
    "boobs", "breast", "penis", "vagina",
    "dick", "pussy", "ass", "fuck",
    "bdsm", "horny", "nsfw", "milf"
]

def contains_nsfw(text):
    pattern = r"\b(" + "|".join(NSFW_KEYWORDS) + r")\b"
    return re.search(pattern, text.lower()) is not None

# ----------------------------------
# Load Model
# ----------------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID
    )

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer

model, tokenizer = load_model()

# ----------------------------------
# Cleanup Function
# ----------------------------------
def clean_joke(text):

    text = re.sub(r"\s+", " ", text).strip()

    # Fix common contractions
    text = re.sub(r"\bWhats\b", "What's", text)
    text = re.sub(r"\bCant\b", "Can't", text)
    text = re.sub(r"\bDont\b", "Don't", text)
    text = re.sub(r"\bIm\b", "I'm", text)

    if text:
        text = text[0].upper() + text[1:]

    # Add question mark if needed
    if text.lower().startswith(
        (
            "what",
            "why",
            "when",
            "where",
            "who",
            "how"
        )
    ) and not text.endswith("?"):
        text += "?"

    elif text and text[-1] not in ".!?":
        text += "."

    return text
def is_bad_joke(text):
    words = text.split()

    if len(words) < 10:
        return True

    if len(words) == 2 and len(set(words)) == 1:
        return True

    bad_endings = [
        "because",
        "and",
        "but",
        "if",
        "the",
        "named",
        "with",
        "for"
    ]

    if any(
        text.lower().endswith(x)
        for x in bad_endings
    ):
        return True

    return False


def score_joke(joke):

    score = 0
    words = joke.split()

    score += min(len(words), 30)

    if "?" in joke:
        score += 10

    if "!" in joke:
        score += 10

    if words and len(set(words)) / len(words) > 0.8:
        score += 20

    return score

# ----------------------------------
# UI Controls
# ----------------------------------
strict_mode = st.toggle(
    "🚫 Strict Mode (Safe Content)",
    value=True
)

topic = st.text_input(
    "📝 Enter a Topic",
    placeholder="e.g. coding, exams, office life, gym"
)

# ----------------------------------
# Joke Style Dropdown
# ----------------------------------
joke_style = st.selectbox(
    "🎭 Select Joke Style",
    [
        "Normal",
        "Roast",
        "Brutal",
        "Sarcastic",
        "Adult",
        "Pun"
    ]
)

STYLE_PREFIX = {
    "Normal": "Funny",
    "Roast": "roast",
    "Brutal": "brutal",
    "Sarcastic": "sarcastic",
    "Adult": "adult humor",
    "Pun":"Pun"
    
}

st.caption(f"Selected Style: {joke_style}")

# ----------------------------------
# Generate Joke
# ----------------------------------
if st.button("🎲 Generate Joke"):

    if not topic.strip():
        st.warning(
            "Please enter a topic first 😅"
        )

    elif strict_mode and contains_nsfw(topic):
        st.error(
            "🚫 Inappropriate topic detected."
        )

    else:
        style = STYLE_PREFIX[joke_style]

        prompt = f"Write a {style} joke about {topic} with a setup and punchline.\nSetup:"

        input_ids = tokenizer.encode(
            prompt,
            return_tensors="pt"
        )

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        with st.spinner("Thinking... Generating jokes"):
            progress = st.progress(0)
            progress.progress(5)
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=120,
                    min_new_tokens=30,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.8,
                    no_repeat_ngram_size=4,
                    num_return_sequences=20
                )
            progress.progress(100)

        jokes = []

        for seq in output:
            text = tokenizer.decode(
                seq,
                skip_special_tokens=True
            )

            text = text.replace(prompt, "").strip()

            if text.startswith("Setup:"):
                text = text[len("Setup:"):].strip()

            if "Punchline:" in text:
                setup, punchline = text.split("Punchline:", 1)
                text = f"Setup: {setup.strip()}\nPunchline: {punchline.strip()}"

            text = clean_joke(text)

            if not is_bad_joke(text):
                jokes.append(text)

        if strict_mode:
            jokes = [j for j in jokes if not contains_nsfw(j)]

        if jokes:
            joke = max(jokes, key=score_joke)

            reactions = [
                "😂",
                "🤣",
                "😆",
                "💀",
                "🔥",
                "🎭"
            ]

            st.success(
                f"Here's your joke {random.choice(reactions)}"
            )

            st.markdown(
                f"""
                <div style="
                    padding:20px;
                    border-radius:10px;
                    border:1px solid #444;
                    font-size:22px;
                ">
                {joke}
                </div>
                """,
                unsafe_allow_html=True
            )

            
