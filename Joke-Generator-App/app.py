import streamlit as st
import torch
import random
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Comedy Courtroom ⚖️", page_icon="🎭")

st.title("⚖️ Comedy Courtroom Simulator")
st.caption("Where jokes are tried, judged, and sentenced.")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Court Registry 📜")
st.sidebar.info("Built by TEJAS OKE & KARTIKEY SINGH")
st.sidebar.markdown("Every joke is a defendant in this courtroom.")

# =========================
# MODEL
# =========================
MODEL_ID = "tejasoke/joke-generator-gpt2"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# =========================
# COURT DATA
# =========================
MUGSHOTS = [
    "https://i.imgur.com/1Q9Z1Zb.png",
    "https://i.imgur.com/8Km9tLL.png",
    "https://i.imgur.com/4M34hi2.png",
    "https://i.imgur.com/6Q7XQ0p.png",
]

CHARGES = [
    "Charged with excessive puns",
    "Suspected of illegal laughter",
    "Arrested for cringe distribution",
    "Convicted of dad-joke violations",
    "Under trial for humor overload",
    "Wanted for emotional giggle damage",
]

VERDICTS = [
    "GUILTY 😂 Sentence: 10 more jokes",
    "INNOCENT… but barely",
    "GUILTY: laughter overload confirmed",
    "CASE DISMISSED (too funny to judge)",
    "MAXIMUM SENTENCE: stand-up comedy career",
]

# =========================
# INPUT SECTION
# =========================
topic = st.text_input("Enter crime (topic):", placeholder="e.g. school, love, coding, etc.")

st.markdown("---")

# =========================
# COURTROOM LAYOUT
# =========================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📂 Defendant File")

with col2:
    st.subheader("⚖️ Trial Chamber")

# =========================
# BUTTON
# =========================
if st.button("Summon Defendant 🎭"):

    if not topic.strip():
        st.warning("Court needs a case to proceed 😅")
    else:

        # loading animation
        loading = st.empty()
        loading.info("📂 Opening criminal joke file...")

        time.sleep(0.7)

        # generate joke
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

        # select random court elements
        mugshot = random.choice(MUGSHOTS)
        charge = random.choice(CHARGES)
        verdict = random.choice(VERDICTS)

        loading.empty()

        # =========================
        # DISPLAY DEFENDANT
        # =========================
        with col1:
            st.image(mugshot, caption="Suspect in custody 📸", width=200)
            st.error(f"⚖️ Charge: {charge}")

        # =========================
        # DISPLAY TRIAL
        # =========================
        with col2:
            st.success("📜 Evidence Submitted")
            st.write(joke)

            st.markdown("---")

            st.subheader("🧑‍⚖️ Judge's Verdict")
            st.warning(verdict)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("⚖️ In this courtroom, humor is both the crime and the punishment.")
