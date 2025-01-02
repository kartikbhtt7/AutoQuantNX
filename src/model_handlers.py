import torch
from transformers import AutoTokenizer, WhisperProcessor
from quantize import ModelHandler
import time

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

    def decode_output(self, outputs):
        predictions = torch.argmax(outputs.logits, dim=2)
        tokens = self.tokenizer.tokenize(self.test_text)
        predicted_labels = [self.original_model.config.id2label[t.item()] for t in predictions[0]]
        
        result = []
        for token, label in zip(tokens, predicted_labels):
            result.append(f"{token:<15} {label}")
        
        return "Tokens and their labels:\n" + "\n".join(result)

class CausalLMHandler(ModelHandler):
    def __init__(self, model_name, model_class, quantization_type, test_text):
        super().__init__(model_name, model_class, quantization_type, test_text)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def run_inference(self, model, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        end_time = time.time()
        return outputs, end_time - start_time

    def decode_output(self, outputs):
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return f"Generated text: {generated_text}"

class EmbeddingModelHandler(ModelHandler):
    def __init__(self, model_name, model_class, quantization_type, test_text):
        super().__init__(model_name, model_class, quantization_type, test_text)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def run_inference(self, model, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time()
        return outputs.last_hidden_state.mean(dim=1), end_time - start_time

    def decode_output(self, outputs):
        return f"Embedding shape: {outputs.shape}"

class WhisperHandler(ModelHandler):
    def __init__(self, model_name, model_class, quantization_type, test_text):
        super().__init__(model_name, model_class, quantization_type, test_text)
        self.processor = WhisperProcessor.from_pretrained(model_name)

    def run_inference(self, model, text):
        inputs = self.processor(text, return_tensors="pt").input_features.to(self.device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(inputs)
        end_time = time.time()
        return outputs, end_time - start_time

    def decode_output(self, outputs):
        return "Whisper model processed successfully"

def get_model_handler(task, model_name, model_class, quantization_type, test_text):
    handler_map = {
        "embedding_finetuning": EmbeddingModelHandler,
        "ner": TokenClassificationHandler,
        "text_classification": SequenceClassificationHandler,
        "whisper_finetuning": WhisperHandler,
        "question_answering": QuestionAnsweringHandler,
        "causal_lm": CausalLMHandler,
    }
    
    handler_class = handler_map.get(task)
    if handler_class:
        return handler_class(model_name, model_class, quantization_type, test_text)
    else:
        raise ValueError(f"No handler found for task: {task}")