import streamlit as st
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(page_title="AI Joke Generator", page_icon="😂")

st.title("😂 AI Joke Generator")
st.caption("Give me a topic, I’ll try to be funny.")
st.markdown("---")
st.caption("Made with 🌚 by TEJAS OKE & KARTIKEY SINGH")

MODEL_ID = "tejasoke/joke-generator-gpt2"

# -------------------------------
# Safer content filter (no slurs)
# -------------------------------
NSFW_KEYWORDS = [
    "sex", "porn", "xxx", "nude", "naked",
    "boobs", "breast", "penis", "vagina",
    "dick", "pussy", "ass", "fuck",
    "bdsm", "horny", "nsfw","gay","milf"
]

def contains_nsfw(text):
    pattern = r"\b(" + "|".join(NSFW_KEYWORDS) + r")\b"
    return re.search(pattern, text.lower()) is not None


# -------------------------------
# Load model once
# -------------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# -------------------------------
# UI Controls
# -------------------------------
strict_mode = st.toggle("🚫 Strict Mode (Block inappropriate content)", value=True)

topic = st.text_input(
    "Enter a topic:",
    placeholder="e.g. office life, coding, exams"
)

# -------------------------------
# Generate Joke
# -------------------------------
if st.button("Generate Joke"):

    if not topic.strip():
        st.warning("Please enter a topic first 😅")

    elif strict_mode and contains_nsfw(topic):
        st.error("🚫 Inappropriate topic detected. Try something cleaner.")

    else:
        input_ids = tokenizer.encode(topic, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=80,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )

        joke = tokenizer.decode(output[0], skip_special_tokens=True)

        # Output filtering
        if strict_mode and contains_nsfw(joke):
            st.warning("⚠️ Generated content was filtered. Regenerating...")

            # Try one more time
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length=80,
                    do_sample=True,
                    temperature=0.8,
                    top_k=40,
                    top_p=0.9,
                    repetition_penalty=1.3,
                    pad_token_id=tokenizer.eos_token_id
                )

            joke = tokenizer.decode(output[0], skip_special_tokens=True)

            if contains_nsfw(joke):
                st.error("❌ Could not generate a safe joke. Try another topic.")
            else:
                st.success("Here’s your joke 🎭")
                st.write(joke)

        else:
            st.success("Here’s your joke 🎭")
            st.write(joke)
