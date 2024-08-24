from flask import Flask, request, jsonify
import torch
import os
from transformers import LlamaForCausalLM, LlamaTokenizer

app = Flask(__name__)

model_name = "meta-llama/Llama-2-7b"
hf_token = os.getenv("HF_TOKEN")
model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=hf_token)


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
