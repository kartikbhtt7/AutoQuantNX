from optimizations.quantize import ModelQuantizer
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ModelHandler:
    """Base class for handling different types of models"""
    
    def __init__(self, model_name, model_class, quantization_type, test_text=None):
        self.model_name = model_name
        self.model_class = model_class
        self.quantization_type = quantization_type
        self.test_text = test_text
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self.original_model = self._load_original_model()
        self.quantized_model = self._load_quantized_model()
    
    def _load_original_model(self):
        """Load the original model"""
        model = self.model_class.from_pretrained(self.model_name)
        return model.to(self.device)
    
    def _load_quantized_model(self):
        """Load the quantized model using ModelQuantizer"""
        model = ModelQuantizer.quantize_model(
            self.model_class, 
            self.model_name, 
            self.quantization_type
        )
        if self.quantization_type not in ["4-bit", "8-bit"]:
            model = model.to(self.device)
        return model

    def _format_metric_value(self, value):
        """Format metric value based on its type"""
        if isinstance(value, (float, np.float32, np.float64)):
            return f"{value:.8f}"
        elif isinstance(value, (int, np.int32, np.int64)):
            return str(value)
        elif isinstance(value, list):
            return "\n" + "\n".join([f"  - {item}" for item in value])
        elif isinstance(value, dict):
            return "\n" + "\n".join([f"  {k}: {v}" for k, v in value.items()])
        else:
            return str(value)

    def run_inference(self, model, text):
        """Run model inference - to be implemented by subclasses"""
        raise NotImplementedError
    
    def decode_output(self, outputs):
        """Decode model outputs - to be implemented by subclasses"""
        raise NotImplementedError

    def compare(self):
        """Compare original and quantized models"""
        try:
            if self.test_text is None:
                logger.warning("No test text provided. Skipping inference testing.")
                return self.quantized_model

            # Run inference
            original_outputs, original_time = self.run_inference(self.original_model, self.test_text)
            quantized_outputs, quantized_time = self.run_inference(self.quantized_model, self.test_text)
            
            original_size = ModelQuantizer.get_model_size(self.original_model)
            quantized_size = ModelQuantizer.get_model_size(self.quantized_model)
            
            logger.info(f"Original Model Size: {original_size:.2f} MB")
            logger.info(f"Quantized Model Size: {quantized_size:.2f} MB")
            logger.info(f"Original Inference Time: {original_time:.4f} seconds")
            logger.info(f"Quantized Inference Time: {quantized_time:.4f} seconds")

            # Compare outputs
            comparison_metrics = self.compare_outputs(
                original_outputs, 
                quantized_outputs
            )
            if comparison_metrics:
                for metric, value in comparison_metrics.items():
                    formatted_value = self._format_metric_value(value)
                    logger.info(f"{metric}: {formatted_value}")

            return self.quantized_model
        except Exception as e:
            logger.error(f"Quantization and comparison failed: {str(e)}")
            raise e
