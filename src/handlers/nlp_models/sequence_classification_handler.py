from ..base_handler import ModelHandler
from transformers import AutoTokenizer
import torch
import time
import numpy as np

class SequenceClassificationHandler(ModelHandler):
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
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        return f"Predicted class: {predicted_class}"

    def compare_outputs(self, original_outputs, quantized_outputs):
        """Compare outputs for sequence classification models"""
        if original_outputs is None or quantized_outputs is None:
            return None
        
        orig_logits = original_outputs.logits.cpu().numpy()
        quant_logits = quantized_outputs.logits.cpu().numpy()
        
        orig_probs = torch.nn.functional.softmax(torch.tensor(orig_logits), dim=-1).numpy()
        quant_probs = torch.nn.functional.softmax(torch.tensor(quant_logits), dim=-1).numpy()
        
        orig_pred = orig_probs.argmax(axis=-1)
        quant_pred = quant_probs.argmax(axis=-1)
        
        metrics = {
            'class_match': float(orig_pred == quant_pred),
            'logits_mse': ((orig_logits - quant_logits) ** 2).mean(),
            'probability_mse': ((orig_probs - quant_probs) ** 2).mean(),
            'max_probability_diff': abs(orig_probs.max() - quant_probs.max()),
            'kl_divergence': float(
                (orig_probs * (np.log(orig_probs + 1e-10) - np.log(quant_probs + 1e-10))).sum()
            )
        }
        
        return metrics