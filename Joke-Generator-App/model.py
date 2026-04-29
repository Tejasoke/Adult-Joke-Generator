from flask import Flask, render_template, request, jsonify
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



print("Current working dir:", os.getcwd())
print("Model path:", os.path.abspath("fine-tuned-gpt2"))
print("Files in model folder:", os.listdir("fine-tuned-gpt2"))

app = Flask(__name__)

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_directory = os.path.join(BASE_DIR, "fine-tuned-gpt2")

# Load model + tokenizer
model = AutoModelForCausalLM.from_pretrained(model_directory)
#tokenizer = AutoTokenizer.from_pretrained(model_directory)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Joke generation route
@app.route('/generate_joke', methods=['POST'])
def generate_joke():
    data = request.get_json()
    topic = data.get('topic', '')

    if not topic.strip():
        return jsonify({'error': 'Please provide a topic'}), 400

    # Encode input
    input_ids = tokenizer.encode(topic, return_tensors='pt')

    # Generate joke
    output = model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        temperature=0.9,
        top_k=50
    )

    joke = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({'joke': joke})

# Run app
if __name__ == '__main__':
    app.run(debug=True)