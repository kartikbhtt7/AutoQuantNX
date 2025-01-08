import torch
from transformers import BitsAndBytesConfig
import logging
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelQuantizer:
    """Handles model quantization and comparison operations"""
    
    @staticmethod
    def quantize_model(model_class, model_name, quantization_type):
        """Quantizes a model based on specified quantization type"""
        try:
            if quantization_type == "4-bit":
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                model = model_class.from_pretrained(model_name, quantization_config=quantization_config)
            elif quantization_type == "8-bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model = model_class.from_pretrained(model_name, quantization_config=quantization_config)
            elif quantization_type == "16-bit-float":
                model = model_class.from_pretrained(model_name)
                model = model.to(torch.float16)
            else:
                raise ValueError(f"Unsupported quantization type: {quantization_type}")
            
            return model
        except Exception as e:
            logger.error(f"Quantization failed: {str(e)}")
            raise

    @staticmethod
    def get_model_size(model):
        """Calculate model size in MB"""
        try:
            torch.save(model.state_dict(), "temp.pth")
            size = os.path.getsize("temp.pth") / (1024 * 1024)
            os.remove("temp.pth")
            return size
        except Exception as e:
            logger.error(f"Failed to get model size: {str(e)}")
            raise

    @staticmethod
    def compare_model_outputs(original_outputs, quantized_outputs):
        """Compare outputs between original and quantized models"""
        try:
            if original_outputs is None or quantized_outputs is None:
                return None

            if hasattr(original_outputs, 'logits') and hasattr(quantized_outputs, 'logits'):
                original_logits = original_outputs.logits.detach().cpu().numpy()
                quantized_logits = quantized_outputs.logits.detach().cpu().numpy()
                
                metrics = {
                    'mse': ((original_logits - quantized_logits) ** 2).mean(),
                    'spearman_corr': spearmanr(original_logits.flatten(), quantized_logits.flatten())[0],
                    'cosine_sim': cosine_similarity(original_logits.reshape(1, -1), quantized_logits.reshape(1, -1))[0][0]
                }
                return metrics
            return None
        except Exception as e:
            logger.error(f"Output comparison failed: {str(e)}")
            raise