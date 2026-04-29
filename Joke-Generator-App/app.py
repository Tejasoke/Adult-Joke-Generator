import streamlit as st
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(page_title="AI Joke Generator", page_icon="😂")
st.sidebar.info("Built by TEJAS OKE & KARTIKEY SINGH")
st.title("😂 AI Joke Generator")
st.caption("Give me a topic, I'll try to be funny.")
st.markdown("---")
st.caption("Made with 🌚 by TEJAS OKE & KARTIKEY SINGH")
MODEL_ID = "tejasoke/joke-generator-gpt2"

# Define explicit image collections for adult jokes
EXPLICIT_IMAGES = [
    "https://i.imgur.com/example1.jpg",  # Replace with actual explicit image URLs
    "https://i.imgur.com/example2.jpg",
    "https://i.imgur.com/example3.jpg",
    # Add more explicit image URLs here
]

BDSM_IMAGES = [
    "https://i.imgur.com/bdsm1.jpg",  # Replace with actual BDSM image URLs
    "https://i.imgur.com/bdsm2.jpg",
    "https://i.imgur.com/bdsm3.jpg",
    # Add more BDSM image URLs here
]

# Function to categorize joke type
def categorize_joke(joke):
    bdsm_keywords = ["bondage", "dominant", "submissive", "master", "slave", "whip", "chains", "cuffs"]
    explicit_keywords = ["fuck", "sex", "naked", "nude", "pussy", "dick", "cock", "ass", "tits"]
    
    joke_lower = joke.lower()
    
    if any(keyword in joke_lower for keyword in bdsm_keywords):
        return "bdsm"
    elif any(keyword in joke_lower for keyword in explicit_keywords):
        return "explicit"
    else:
        return "regular"

# Load model once (safe + stable)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# Input
topic = st.text_input(
    "Enter a topic:",
    placeholder="e.g. any topic including adult words"
)

# Generate joke
if st.button("Generate Joke"):
    if not topic.strip():
        st.warning("Please enter a topic first 😅")
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
        
        # Categorize joke and select appropriate image
        joke_category = categorize_joke(joke)
        
        if joke_category == "bdsm":
            selected_image = random.choice(BDSM_IMAGES)
        elif joke_category == "explicit":
            selected_image = random.choice(EXPLICIT_IMAGES)
        else:
            # Default to a generic funny image for non-explicit jokes
            selected_image = "https://media.giphy.com/media/3o7aD2saalBwwftBIY/giphy.gif"

        st.success("Here's your joke 🎭")
        st.write(joke)
        st.image(selected_image, width=400)
