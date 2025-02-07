from .base_handler import ModelHandler
from .nlp_models.sequence_classification_handler import SequenceClassificationHandler
from .nlp_models.question_answering_handler import QuestionAnsweringHandler
from .nlp_models.token_classification_handler import TokenClassificationHandler
from .nlp_models.causal_lm_handler import CausalLMHandler
from .nlp_models.embedding_model_handler import EmbeddingModelHandler
from .audio_models.whisper_handler import WhisperHandler
from .masked_lm_handler import MaskedLMHandler
from .seq2seq_lm_handler import Seq2SeqLMHandler
from .multiple_choice_handler import MultipleChoiceHandler
from .img_models.image_classification_handler import ImageClassificationHandler

from transformers import (
    AutoModel,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForMultipleChoice,
)

TASK_CONFIGS = {
    "embedding_finetuning": {
        "model_class": AutoModel,
        "handler_class": EmbeddingModelHandler,
        "example_text": "Hey, I am feeling way to good to be true.",
    },
    "ner": {
        "model_class": AutoModelForTokenClassification,
        "handler_class": TokenClassificationHandler,
        "example_text": "John works at Google in New York as a software engineer.",
    },
    "text_classification": {
        "model_class": AutoModelForSequenceClassification,
        "handler_class": SequenceClassificationHandler,
        "example_text": "This movie was great and I loved it.",
    },
    "question_answering": {
        "model_class": AutoModelForQuestionAnswering,
        "handler_class": QuestionAnsweringHandler,
        "example_text": "The pyramids were built in ancient Egypt. QUES: When were the pyramids built?",
    },
    "causal_lm": {
        "model_class": AutoModelForCausalLM,
        "handler_class": CausalLMHandler,
        "example_text": "Once upon a time, there was ",
    },
    "mask_lm": {
        "model_class": AutoModelForMaskedLM,
        "handler_class": MaskedLMHandler,
        "example_text": "The quick brown [MASK] jumps over the lazy dog.",
    },
    "seq2seq_lm": {
        "model_class": AutoModelForSeq2SeqLM,
        "handler_class": Seq2SeqLMHandler,
        "example_text": "Translate English to French: The house is wonderful.",
    },
    "multiple_choice": {
        "model_class": AutoModelForMultipleChoice,
        "handler_class": MultipleChoiceHandler,
        "example_text": "What is the capital of France? (A) Paris (B) London (C) Berlin (D) Rome",
    },
    "whisper_finetuning": {
        "model_class": None, # Not implemented
        "handler_class": WhisperHandler,
        "example_text": "!!!!!NOT IMPLEMENTED!!!!!",
    },
    "image_classification": {
        "model_class": None,  # Not implemented
        "handler_class": ImageClassificationHandler,
        "example_text": "!!!!!NOT IMPLEMENTED!!!!!",
    },
}

def get_model_handler(task: str, model_name: str, quantization_type: str, test_text: str):
    task_config = TASK_CONFIGS.get(task)
    if not task_config:
        raise ValueError(f"No configuration found for task: {task}")

    handler_class = task_config["handler_class"]
    model_class = task_config["model_class"]
    return handler_class(model_name, model_class, quantization_type, test_text)