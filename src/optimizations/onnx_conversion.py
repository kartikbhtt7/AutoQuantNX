import os
from transformers import AutoTokenizer, WhisperProcessor, AutoFeatureExtractor
from optimum.onnxruntime import (
    ORTModelForQuestionAnswering,
    ORTModelForCausalLM,
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
    ORTModelForSpeechSeq2Seq,
    ORTOptimizer,
    ORTModelForMaskedLM,
    ORTModelForSeq2SeqLM,
    ORTModelForMultipleChoice,
    ORTModelForImageClassification,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK_MAPPING = {
    # NLP models
    "ner": (ORTModelForTokenClassification, AutoTokenizer),
    "text_classification": (ORTModelForSequenceClassification, AutoTokenizer),
    "question_answering": (ORTModelForQuestionAnswering, AutoTokenizer),
    "causal_lm": (ORTModelForCausalLM, AutoTokenizer),
    "mask_lm": (ORTModelForMaskedLM, AutoTokenizer),
    "seq2seq_lm": (ORTModelForSeq2SeqLM, AutoTokenizer),
    "multiple_choice": (ORTModelForMultipleChoice, AutoTokenizer),
    # Audio models
    "whisper_finetuning": (ORTModelForSpeechSeq2Seq, WhisperProcessor),
    # Vision models
    "image_classification": (ORTModelForImageClassification, AutoFeatureExtractor),
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

    ORTModelClass, ProcessorClass = TASK_MAPPING[task]

    try:
        if task == "embedding_finetuning":
            ort_optimizer = ORTOptimizer.from_pretrained(model_name)
            ort_optimizer.export(output_dir=output_dir, task="feature-extraction")
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
