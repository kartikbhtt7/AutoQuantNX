from ..base_handler import ModelHandler
from transformers import AutoTokenizer
import torch
import time

class TokenClassificationHandler(ModelHandler):
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

    def decode_output(self, model, outputs):
        tokens = self.tokenizer.convert_ids_to_tokens(outputs['input_ids'][0])
        labels = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        decoded_labels = [model.config.id2label[label] for label in labels]
        return dict(zip(tokens, decoded_labels))

    def compare_outputs(self, original_outputs, quantized_outputs):
        """Compare outputs for token classification models"""
        if original_outputs is None or quantized_outputs is None:
            return None
        
        orig_logits = original_outputs.logits.cpu().numpy()
        quant_logits = quantized_outputs.logits.cpu().numpy()
        
        orig_preds = orig_logits.argmax(axis=-1)
        quant_preds = quant_logits.argmax(axis=-1)
        
        input_tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer(self.test_text, return_tensors='pt')['input_ids'][0]
        )
        
        orig_labels = [self.original_model.config.id2label[p] for p in orig_preds[0]]
        quant_labels = [self.quantized_model.config.id2label[p] for p in quant_preds[0]]
        
        original_results = list(zip(input_tokens, orig_labels))
        quantized_results = list(zip(input_tokens, quant_labels))
        
        token_matches = sum(o_label == q_label for o_label, q_label in zip(orig_labels, quant_labels))
        total_tokens = len(orig_labels)
        
        metrics = {
            'original_predictions': original_results,
            'quantized_predictions': quantized_results,
            'token_level_accuracy': float(token_matches) / total_tokens if total_tokens > 0 else 0.0,
            'sequence_exact_match': float((orig_preds == quant_preds).all()),
            'logits_mse': ((orig_logits - quant_logits) ** 2).mean(),
        }
        
        return metrics