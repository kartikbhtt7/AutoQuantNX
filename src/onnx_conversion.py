import os
from transformers import AutoTokenizer, WhisperProcessor
from optimum.onnxruntime import (
    ORTModelForQuestionAnswering,
    ORTModelForCausalLM,
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
    ORTModelForSpeechSeq2Seq,
    ORTModelForMaskedLM,
    ORTModelForImageClassification,
    ORTModelForSeq2SeqLM,
    ORTOptimizer,
)
import onnx
from onnxconverter_common import float16
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK_MAPPING = {
    "embedding_finetuning": (None, AutoTokenizer, "feature-extraction"),
    "ner": (ORTModelForTokenClassification, AutoTokenizer, None),
    "text_classification": (ORTModelForSequenceClassification, AutoTokenizer, None),
    "whisper_finetuning": (ORTModelForSpeechSeq2Seq, WhisperProcessor, None),
    "question_answering": (ORTModelForQuestionAnswering, AutoTokenizer, None),
    "causal_lm": (ORTModelForCausalLM, AutoTokenizer, None),
    "masked_lm": (ORTModelForMaskedLM, AutoTokenizer, None),
    "image_classification": (ORTModelForImageClassification, AutoTokenizer, None),
    "seq2seq": (ORTModelForSeq2SeqLM, AutoTokenizer, None),
}

def optimize_onnx_model(model_path):
    """Optimize ONNX model for inference"""
    try:
        from onnxruntime.transformers import optimizer
        opt_model = optimizer.optimize_model(
            model_path,
            model_type='bert',
            num_heads=12,
            hidden_size=768
        )
        opt_model.save_model_to_file(model_path)
        logger.info(f"Model optimized successfully: {model_path}")
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")

def convert_to_onnx(model_name, task, output_dir, optimization_level="O1"):
    """
    Convert model to ONNX format with specified optimization level
    """
    logger.info(f"Converting model: {model_name} for task: {task}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if task not in TASK_MAPPING:
        logger.error(f"Task {task} is not supported for ONNX conversion")
        return None

    ORTModelClass, ProcessorClass, special_task = TASK_MAPPING[task]

    try:
        if task == "embedding_finetuning":
            ort_optimizer = ORTOptimizer.from_pretrained(
                model_name,
                feature=special_task,
                optimization_level=optimization_level
            )
            ort_optimizer.export(output_dir=output_dir)
        else:
            # Use optimum's default optimization settings based on level
            ort_model = ORTModelClass.from_pretrained(
                model_name,
                export=True,
                optimize=True,
                optimization_level=optimization_level
            )
            ort_model.save_pretrained(output_dir)

        processor = ProcessorClass.from_pretrained(model_name)
        processor.save_pretrained(output_dir)
        
        # Optimize the exported model
        model_files = [f for f in os.listdir(output_dir) if f.endswith('.onnx')]
        for model_file in model_files:
            model_path = os.path.join(output_dir, model_file)
            optimize_onnx_model(model_path)
            
        logger.info(f"Conversion complete. Model saved to: {output_dir}")
        return output_dir
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        return None

def quantize_onnx_model(model_dir, quantization_type):
    """Quantize ONNX model with advanced configuration"""
    for filename in os.listdir(model_dir):
        if filename.endswith('.onnx'):
            input_model_path = os.path.join(model_dir, filename)
            output_model_path = os.path.join(model_dir, f"quantized_{filename}")
            
            try:
                model = onnx.load(input_model_path)
                
                if quantization_type == "16-bit-float":
                    model_fp16 = float16.convert_float_to_float16(
                        model,
                        keep_io_types=True,
                        disable_shape_infer=False
                    )
                    onnx.save(model_fp16, output_model_path)
                    
                elif quantization_type in ["8-bit", "16-bit-int"]:
                    from onnxruntime.quantization import (
                        quantize_dynamic,
                        QuantizationMode,
                        QuantType
                    )
                    
                    quant_type_mapping = {
                        "8-bit": QuantType.QInt8,
                        "16-bit-int": QuantType.QInt16,
                    }
                    
                    quantize_dynamic(
                        model_input=input_model_path,
                        model_output=output_model_path,
                        weight_type=quant_type_mapping[quantization_type],
                        optimize_model=True,
                        per_channel=True,
                        reduce_range=True,
                        mode=QuantizationMode.IntegerOps
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

def push_onnx_to_hub(api, local_path, repo_name, commit_message=None):
    """Push ONNX model to HuggingFace Hub with metadata"""
    try:
        api.create_repo(repo_id=repo_name, exist_ok=True)
        
        # Create model card with ONNX details
        model_card = f"""---
tags:
- onnx
- optimum
library_name: optimum
---

# ONNX Model - {repo_name}

This model has been converted to ONNX format using the Optimum library.

## Model Details
- Original model: {repo_name}
- Format: ONNX
- Optimizations: Contains both original and quantized versions
"""
        
        with open(os.path.join(local_path, "README.md"), "w") as f:
            f.write(model_card)

        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_name,
            repo_type="model",
            commit_message=commit_message or "Upload ONNX model"
        )
        
        logger.info(f"ONNX model pushed to Hub: {repo_name}")
        return repo_name
        
    except Exception as e:
        logger.error(f"Error pushing to Hub: {str(e)}")
        return None