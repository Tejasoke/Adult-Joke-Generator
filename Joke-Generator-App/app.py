import streamlit as st
import torch
import random
import requests
from bs4 import BeautifulSoup
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(page_title="AI Joke Generator", page_icon="😂")
st.sidebar.info("Built by TEJAS OKE & KARTIKEY SINGH")
st.title("😂 AI Joke Generator")
st.caption("Give me a topic, I'll try to be funny.")
st.markdown("---")
st.caption("Made with 🌚 by TEJAS OKE & KARTIKEY SINGH")
MODEL_ID = "tejasoke/joke-generator-gpt2"

# Function to get random explicit image URL
def get_random_explicit_image():
    # Method 1: Using random image from Imgur (explicit category)
    try:
        # Using Imgur's gallery endpoint for random images
        # Note: This is a simplified approach - actual implementation might need OAuth
        imgur_url = "https://api.imgur.com/3/gallery/random/viral/0.json"
        headers = {"Authorization": "Client-ID YOUR_CLIENT_ID"}  # You'd need to register for an API key
        
        # Try without auth first (public images)
        response = requests.get("https://imgur.com/search/score?q=nsfw", headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            images = soup.find_all('img')
            if images:
                # Get random image from the results
                img_url = random.choice(images).get('src')
                if img_url and not img_url.startswith('http'):
                    img_url = 'https:' + img_url
                return img_url
    except Exception as e:
        st.error(f"Error fetching image: {str(e)}")
    
    # Method 2: Fallback to predefined explicit image URLs
    fallback_urls = [
        "https://i.imgur.com/random.jpg",  # Imgur's random image endpoint
        "https://picsum.photos/seed/explicit/800/600.jpg",  # Random placeholder
        "https://source.unsplash.com/featured/?nsfw",  # Unsplash with NSFW tag
    ]
    return random.choice(fallback_urls)

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
        
        # Get random explicit image
        selected_image = get_random_explicit_image()

        st.success("Here's your joke 🎭")
        st.write(joke)
        st.image(selected_image, width=400)
