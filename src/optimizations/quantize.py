import os
import torch
import onnx
import logging
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BitsAndBytesConfig
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType

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

def quantize_onnx_model(model_dir, quantization_type):
    """
    Quantize ONNX model in the specified directory.
    """
    logger.info(f"Quantizing ONNX model in: {model_dir}")
    for filename in os.listdir(model_dir):
        if filename.endswith('.onnx'):
            input_model_path = os.path.join(model_dir, filename)
            output_model_path = os.path.join(model_dir, f"quantized_{filename}")

            try:
                model = onnx.load(input_model_path)

                if quantization_type == "16-bit-float":
                    model_fp16 = float16.convert_float_to_float16(model)
                    onnx.save(model_fp16, output_model_path)
                elif quantization_type in ["8-bit", "16-bit-int"]:
                    quant_type_mapping = {
                        "8-bit": QuantType.QInt8,
                        "16-bit-int": QuantType.QInt16,
                    }
                    quantize_dynamic(
                        model_input=input_model_path,
                        model_output=output_model_path,
                        weight_type=quant_type_mapping[quantization_type]
                    )
                else:
                    logger.error(f"Unsupported quantization type: {quantization_type}")
                    continue

                os.remove(input_model_path)
                os.rename(output_model_path, input_model_path)

                logger.info(f"Quantized ONNX model saved to: {input_model_path}")
            except Exception as e:
                logger.error(f"Error during ONNX quantization: {str(e)}")
                if os.path.exists(output_model_path):
                    os.remove(output_model_path)