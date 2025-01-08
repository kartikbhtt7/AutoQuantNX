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

    def compare_outputs(self, original_outputs, quantized_outputs):
        """Compare outputs for question answering models"""
        if original_outputs is None or quantized_outputs is None:
            return None
        
        orig_start = original_outputs.start_logits.cpu().numpy()
        orig_end = original_outputs.end_logits.cpu().numpy()
        quant_start = quantized_outputs.start_logits.cpu().numpy()
        quant_end = quantized_outputs.end_logits.cpu().numpy()
        
        orig_start_pos = orig_start.argmax()
        orig_end_pos = orig_end.argmax()
        quant_start_pos = quant_start.argmax()
        quant_end_pos = quant_end.argmax()

        input_ids = self.tokenizer.encode(self.test_text)
        original_answer = self.tokenizer.decode(input_ids[orig_start_pos:orig_end_pos + 1])
        quantized_answer = self.tokenizer.decode(input_ids[quant_start_pos:quant_end_pos + 1])
            
        metrics = {
            'original_answer': original_answer,
            'quantized_answer': quantized_answer,
            'start_position_match': float(orig_start_pos == quant_start_pos),
            'end_position_match': float(orig_end_pos == quant_end_pos),
            'start_logits_mse': ((orig_start - quant_start) ** 2).mean(),
            'end_logits_mse': ((orig_end - quant_end) ** 2).mean(),
        }
    
        return metrics