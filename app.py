from flask import Flask, request, jsonify
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

app = Flask(__name__)

model = LlamaForCausalLM.from_pretrained("facebook/llama-7b")
tokenizer = LlamaTokenizer.from_pretrained("facebook/llama-7b")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_data = data.get("user_data", "")

    inputs = tokenizer.encode(user_data, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50)
    routine = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"routine": routine})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
