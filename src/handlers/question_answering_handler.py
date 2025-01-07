from .base_handler import ModelHandler
from transformers import AutoTokenizer
import torch
import time

class QuestionAnsweringHandler(ModelHandler):
    def __init__(self, model_name, model_class, quantization_type, test_text):
        super().__init__(model_name, model_class, quantization_type, test_text)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def run_inference(self, model, text):
        parts = text.split('QUES')
        context = parts[0].strip()
        question = parts[1].strip()
        inputs = self.tokenizer(question, context, return_tensors='pt', truncation=True, padding=True).to(self.device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time()
        return outputs, end_time - start_time

    def decode_output(self, outputs):
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        answer_start = torch.argmax(start_logits)
        answer_end = torch.argmax(end_logits) + 1
        input_ids = self.tokenizer.encode(self.test_text)
        answer = self.tokenizer.decode(input_ids[answer_start:answer_end])
        return f"Answer: {answer}"
