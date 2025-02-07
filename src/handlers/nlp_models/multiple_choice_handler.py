from ..base_handler import ModelHandler
from transformers import AutoTokenizer
import torch
import time
import numpy as np

class MultipleChoiceHandler(ModelHandler):
    def __init__(self, model_name, model_class, quantization_type, test_text):
        super().__init__(model_name, model_class, quantization_type, test_text)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def run_inference(self, model, text):
        choices = [text.split(f"({chr(65 + i)})")[1].strip() for i in range(4)]
        inputs = self.tokenizer(choices, return_tensors='pt', padding=True).to(self.device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time()
        return outputs, end_time - start_time

    def decode_output(self, outputs):
        logits = outputs.logits
        predicted_choice = chr(65 + logits.argmax().item())
        return f"Predicted choice: {predicted_choice}"

    def compare_outputs(self, original_outputs, quantized_outputs):
        if original_outputs is None or quantized_outputs is None:
            return None

        original_logits = original_outputs.logits.detach().cpu().numpy()
        quantized_logits = quantized_outputs.logits.detach().cpu().numpy()

        metrics = {
            'mse': ((original_logits - quantized_logits) ** 2).mean(),
            'top_1_accuracy': np.mean(
                np.argmax(original_logits, axis=-1) == np.argmax(quantized_logits, axis=-1)
            ),
        }
        return metrics