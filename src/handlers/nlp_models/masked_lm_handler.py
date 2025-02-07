from ..base_handler import ModelHandler
from transformers import AutoTokenizer
import torch
import time
import numpy as np

class MaskedLMHandler(ModelHandler):
    def __init__(self, model_name, model_class, quantization_type, test_text):
        super().__init__(model_name, model_class, quantization_type, test_text)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def run_inference(self, model, text):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time()
        return outputs, inputs, end_time - start_time

    def decode_output(self, outputs, inputs):
        logits = outputs.logits
        masked_index = torch.where(inputs['input_ids'] == self.tokenizer.mask_token_id)[1]
        predicted_token_id = logits[0, masked_index].argmax(axis=-1)
        return self.tokenizer.decode(predicted_token_id)

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