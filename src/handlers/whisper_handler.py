from .base_handler import ModelHandler
from transformers import WhisperProcessor
import torch
import time

class WhisperHandler(ModelHandler):
    def __init__(self, model_name, model_class, quantization_type, test_text):
        super().__init__(model_name, model_class, quantization_type, test_text)
        self.processor = WhisperProcessor.from_pretrained(model_name)

    def run_inference(self, model, audio_input):
        inputs = self.processor(audio_input, return_tensors="pt").to(self.device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs)
        end_time = time.time()
        return outputs, end_time - start_time

    def decode_output(self, outputs):
        return self.processor.batch_decode(outputs, skip_special_tokens=True)
