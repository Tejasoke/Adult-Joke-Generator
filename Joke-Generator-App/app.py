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

# Random explicit image URLs (replace with actual URLs)
EXPLICIT_IMAGES = [
    "https://i.imgur.com/explicit1.jpg",
    "https://i.imgur.com/explicit2.jpg",
    "https://i.imgur.com/explicit3.jpg",
    "https://i.imgur.com/explicit4.jpg",
    "https://i.imgur.com/explicit5.jpg",
    # Add more URLs as needed
]

# Load model once
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
        
        # Select random explicit image
        selected_image = random.choice(EXPLICIT_IMAGES)

        st.success("Here's your joke 🎭")
        st.write(joke)
        st.image(selected_image, width=400)
