from .base_handler import ModelHandler
from transformers import AutoTokenizer
import torch
import time
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CausalLMHandler(ModelHandler):
    def __init__(self, model_name, model_class, quantization_type, test_text):
        super().__init__(model_name, model_class, quantization_type, test_text)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def run_inference(self, model, text):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        end_time = time.time()
        return outputs, end_time - start_time

    def decode_output(self, outputs):
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def compare_outputs(self, original_outputs, quantized_outputs):
        """Compare outputs for causal language models"""
        if original_outputs is None or quantized_outputs is None:
            return None
            
        original_tokens = original_outputs[0].cpu().numpy()
        quantized_tokens = quantized_outputs[0].cpu().numpy()
        
        metrics = {
            'sequence_similarity': np.mean(original_tokens == quantized_tokens),
            'sequence_length_diff': abs(len(original_tokens) - len(quantized_tokens)),
            'vocab_distribution_correlation': spearmanr(
                np.bincount(original_tokens), 
                np.bincount(quantized_tokens)
            )[0] if len(original_tokens) == len(quantized_tokens) else 0.0
        }
        
        original_text = self.decode_output(original_outputs)
        quantized_text = self.decode_output(quantized_outputs)
        metrics['decoded_text_match'] = float(original_text == quantized_text)
        metrics['original_model_text'] = original_text
        metrics['quantized_model_text'] = quantized_text
        return metrics