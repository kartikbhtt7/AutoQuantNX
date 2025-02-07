from ..base_handler import ModelHandler
from transformers import AutoTokenizer
import torch
import time
import numpy as np

class Seq2SeqLMHandler(ModelHandler):
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
        if original_outputs is None or quantized_outputs is None:
            return None

        original_tokens = original_outputs[0].cpu().numpy()
        quantized_tokens = quantized_outputs[0].cpu().numpy()

        metrics = {
            'sequence_similarity': np.mean(original_tokens == quantized_tokens),
            'sequence_length_diff': abs(len(original_tokens) - len(quantized_tokens)),
        }
        return metrics