
import os
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from core.model import DGESimpleTransformer
from utils.model_manager import ModelManager
from transformers import GPT2Tokenizer
import threading

# V 0.17.0: Remote Inference Server
# Allows live chat with the model currently active on the RunPod instance.

app = Flask(__name__)
mgr = ModelManager()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global state to hold the "live" model and tokenizer
active_state = {
    "model": None,
    "tokenizer": None,
    "lock": threading.Lock()
}

def load_live_model(family=None, stage=None):
    """Loads the latest model for remote inference."""
    with active_state["lock"]:
        try:
            # If no specific model requested, find the latest in the directory
            if not family or not stage:
                models = mgr.list_models()
                if not models:
                    return False, "No models found on-pod."
                meta = models[-1] # Pick most recent
                family, stage = meta['family'], meta['stage']
            
            state_dict, config = mgr.load_stage(family, stage, device=device)
            model = DGESimpleTransformer(
                vocab_size=config['vocab_size'],
                d_model=config['d_model'],
                n_layer=config['n_layer'],
                n_head=config['n_head']
            )
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            
            active_state["model"] = model
            if active_state["tokenizer"] is None:
                active_state["tokenizer"] = GPT2Tokenizer.from_pretrained('gpt2')
                active_state["tokenizer"].pad_token = active_state["tokenizer"].eos_token
            
            return True, f"Loaded {family}/{stage}."
        except Exception as e:
            return False, str(e)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 50)
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
        
    with active_state["lock"]:
        if active_state["model"] is None:
            # Try to auto-load latest model
            success, msg = load_live_model()
            if not success:
                return jsonify({"error": f"Model not loaded: {msg}"}), 500
        
        tokenizer = active_state["tokenizer"]
        model = active_state["model"]
        
        try:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                generated = input_ids
                for _ in range(max_tokens):
                    logits, _ = model(generated)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            
            response_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            return jsonify({"response": response_text})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    with active_state["lock"]:
        return jsonify({
            "model_ready": active_state["model"] is not None,
            "device": str(device)
        })

if __name__ == "__main__":
    # Pre-load latest model if possible
    load_live_model()
    # Run on port 5000 (standard for local dev)
    # RunPod will map this via HTTP/TCP mapping
    app.run(host='0.0.0.0', port=5000)
