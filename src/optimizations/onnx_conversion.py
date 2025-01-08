import os
from transformers import AutoTokenizer, WhisperProcessor
from optimum.onnxruntime import (
    ORTModelForQuestionAnswering,
    ORTModelForCausalLM,
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
    ORTModelForSpeechSeq2Seq,
    ORTOptimizer,
)
import onnx
from onnxconverter_common import float16
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK_MAPPING = {
    "ner": (ORTModelForTokenClassification, AutoTokenizer, None),
    "text_classification": (ORTModelForSequenceClassification, AutoTokenizer, None),
    "whisper_finetuning": (ORTModelForSpeechSeq2Seq, WhisperProcessor, None),
    "question_answering": (ORTModelForQuestionAnswering, AutoTokenizer, None),
    "causal_lm": (ORTModelForCausalLM, AutoTokenizer, None),
}

def convert_to_onnx(model_name, task, output_dir):
    """
    Convert model to ONNX format for the specified task.
    """
    logger.info(f"Converting model: {model_name} for task: {task}")
    
    os.makedirs(output_dir, exist_ok=True)

    if task not in TASK_MAPPING:
        logger.error(f"Task {task} is not supported for ONNX conversion in this script.")
        return None

    ORTModelClass, ProcessorClass, special_task = TASK_MAPPING[task]

    try:
        if task == "embedding_finetuning":
            ort_optimizer = ORTOptimizer.from_pretrained(model_name)
            ort_optimizer.export(output_dir=output_dir, task=special_task)
        else:
            ort_model = ORTModelClass.from_pretrained(model_name, export=True)
            ort_model.save_pretrained(output_dir)

        processor = ProcessorClass.from_pretrained(model_name)
        processor.save_pretrained(output_dir)

        logger.info(f"Conversion complete. Model saved to: {output_dir}")
        return output_dir
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        return None

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
                    from onnxruntime.quantization import quantize_dynamic, QuantType

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
