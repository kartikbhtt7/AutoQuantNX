import torch

class ModelHandler:
    def __init__(self, model_name, model_class, quantization_type, test_text):
        self.model_name = model_name
        self.model_class = model_class
        self.quantization_type = quantization_type
        self.test_text = test_text
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.original_model = self.model_class.from_pretrained(self.model_name).to(self.device)
        self.quantized_model = None

    def quantize(self):
        pass

    def run_inference(self, model, text):
        pass

    def decode_output(self, outputs):
        pass
