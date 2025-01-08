from .base_handler import ModelHandler
from transformers import AutoTokenizer
import torch
import time
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingModelHandler(ModelHandler):
    def __init__(self, model_name, model_class, quantization_type, test_text):
        super().__init__(model_name, model_class, quantization_type, test_text)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def run_inference(self, model, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time()
        return outputs, end_time - start_time

    def decode_output(self, outputs):
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def compare_outputs(self, original_outputs, quantized_outputs):
        """Compare outputs for embedding models"""
        if original_outputs is None or quantized_outputs is None:
            return None
            
        original_embeds = original_outputs.last_hidden_state.cpu().numpy()
        quantized_embeds = quantized_outputs.last_hidden_state.cpu().numpy()
        
        metrics = {
            'mse': ((original_embeds - quantized_embeds) ** 2).mean(),
            'cosine_similarity': cosine_similarity(
                original_embeds.reshape(1, -1), 
                quantized_embeds.reshape(1, -1)
            )[0][0],
            'correlation': spearmanr(
                original_embeds.flatten(), 
                quantized_embeds.flatten()
            )[0],
            'norm_difference': np.abs(
                np.linalg.norm(original_embeds) - 
                np.linalg.norm(quantized_embeds)
            )
        }
        
        return metrics