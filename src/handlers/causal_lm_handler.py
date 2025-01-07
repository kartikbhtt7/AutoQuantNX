from .base_handler import ModelHandler
from transformers import AutoTokenizer
import torch
import time

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
