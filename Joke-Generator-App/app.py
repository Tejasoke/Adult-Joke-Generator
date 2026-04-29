import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(page_title="AI Joke Generator", page_icon="😂")

st.title("😂 AI Joke Generator")
st.caption("Give me a topic, I’ll try to be funny.")

MODEL_NAME = "Tejasoke/joke-generator-gpt2"  # 🔥 your Hugging Face repo

# Load model (cached)
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# Input
topic = st.text_input("Enter a topic:", placeholder="e.g. programming, exams, cats")

# Controls (🔥 makes your app feel pro)
temperature = st.slider("Creativity", 0.5, 1.5, 0.9)
max_len = st.slider("Joke Length", 20, 100, 60)

# Generate
if st.button("Generate Joke"):
    if topic.strip() == "":
        st.warning("Please enter a topic first 😅")
    else:
        with st.spinner("Cooking a joke... 🍳"):
            input_ids = tokenizer.encode(topic, return_tensors="pt")

            output = model.generate(
                input_ids,
                max_length=max_len,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )

            joke = tokenizer.decode(output[0], skip_special_tokens=True)

        st.success("Here’s your joke 🎭")
        st.write(joke)
