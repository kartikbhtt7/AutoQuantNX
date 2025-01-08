from .base_handler import ModelHandler
from .sequence_classification_handler import SequenceClassificationHandler
from .question_answering_handler import QuestionAnsweringHandler
from .token_classification_handler import TokenClassificationHandler
from .causal_lm_handler import CausalLMHandler
from .embedding_model_handler import EmbeddingModelHandler
from .whisper_handler import WhisperHandler

from transformers import (
    AutoModel,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    WhisperForConditionalGeneration,
)

TASK_CONFIGS = {
    "embedding_finetuning": {
        "model_class": AutoModel,
        "handler_class": EmbeddingModelHandler,
        "example_text": "This is amazing!",
    },
    "ner": {
        "model_class": AutoModelForTokenClassification,
        "handler_class": TokenClassificationHandler,
        "example_text": "John works at Google in New York",
    },
    "text_classification": {
        "model_class": AutoModelForSequenceClassification,
        "handler_class": SequenceClassificationHandler,
        "example_text": "This movie was great!",
    },
    "whisper_finetuning": {
        "model_class": WhisperForConditionalGeneration,
        "handler_class": WhisperHandler,
        "example_text": "!!!WORKING ON THIS!!!",
    },
    "question_answering": {
        "model_class": AutoModelForQuestionAnswering,
        "handler_class": QuestionAnsweringHandler,
        "example_text": "The pyramids were built in ancient Egypt. QUES: When were the pyramids built?",
    },
    "causal_lm": {
        "model_class": AutoModelForCausalLM,
        "handler_class": CausalLMHandler,
        "example_text": "Once upon a time",
    },
}

def get_model_handler(task: str, model_name: str, quantization_type: str, test_text: str):
    task_config = TASK_CONFIGS.get(task)
    if not task_config:
        raise ValueError(f"No configuration found for task: {task}")

    handler_class = task_config["handler_class"]
    model_class = task_config["model_class"]
    return handler_class(model_name, model_class, quantization_type, test_text)