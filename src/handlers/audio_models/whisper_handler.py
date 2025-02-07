from ..base_handler import ModelHandler

class WhisperHandler(ModelHandler):
    def __init__(self, model_name, model_class, quantization_type, test_text):
        super().__init__(model_name, model_class, quantization_type, test_text)

    def run_inference(self, model, text):
        raise NotImplementedError("Image classification is not implemented.")

    def decode_output(self, outputs):
        raise NotImplementedError("Image classification is not implemented.")

    def compare_outputs(self, original_outputs, quantized_outputs):
        raise NotImplementedError("Image classification is not implemented.")