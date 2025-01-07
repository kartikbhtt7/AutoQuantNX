import gradio as gr
import torch
from transformers import (
    AutoModel,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    WhisperForConditionalGeneration,
)
from huggingface_hub import HfApi
import logging
from typing import Tuple, Dict, Any
from src.utilities.utilities import ResourceManager, push_to_hub
from src.model_handlers_temp import get_model_handler
from onnx_conversion import convert_to_onnx, quantize_onnx_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "embedding_finetuning": AutoModel,
    "ner": AutoModelForTokenClassification,
    "text_classification": AutoModelForSequenceClassification,
    "whisper_finetuning": WhisperForConditionalGeneration,
    "question_answering": AutoModelForQuestionAnswering,
    "causal_lm": AutoModelForCausalLM,
}

EXAMPLE_TEXTS = {
    "text_classification": "This movie was great!",
    "ner": "John works at Google in New York",
    "question_answering": "Context: The pyramids were built in ancient Egypt. Q: When were the pyramids built?",
    "causal_lm": "Once upon a time",
    "embedding_finetuning": "This is amazing!",
    "whisper_finetuning": "!!!WORKING ON THIS!!!",
}

def process_model(
    model_name: str,
    task: str,
    quantization_type: str,
    enable_onnx: bool,
    onnx_quantization: str,
    hf_token: str,
    repo_name: str,
    test_text: str
) -> Tuple[Dict[str, Any], str]:
    try:
        resource_manager = ResourceManager()
        status_updates = []
        status = {
            "status": "Processing",
            "progress": 0,
            "current_step": "Initializing",
        }

        if not model_name or not hf_token or not repo_name:
            return {
                "status": "Error",
                "progress": 0,
                "current_step": "Validation Failed",
            }, "Model name, HuggingFace token, and repository name are required."

        status["progress"] = 0.2
        status["current_step"] = "Initialization"
        status_updates.append("Initialization complete")

        quantized_model_path = None

        if quantization_type != "None":
            status.update({"progress": 0.4, "current_step": "Quantization"})
            status_updates.append(f"Applying {quantization_type} quantization")

            if not test_text:
                test_text = EXAMPLE_TEXTS.get(task, "Sample text for testing")

            try:
                model_class = MODEL_CLASSES[task]
                handler = get_model_handler(task, model_name, model_class, quantization_type, test_text)
                quantized_model = handler.quantize_and_compare()

                quantized_model_path = str(resource_manager.temp_dirs["quantized"] / "model")
                quantized_model.save_pretrained(quantized_model_path)
                status_updates.append("Quantization completed successfully")
            except Exception as e:
                logger.error(f"Quantization error: {str(e)}", exc_info=True)
                return {
                    "status": "Error",
                    "progress": 0.4,
                    "current_step": "Quantization Failed",
                }, f"Quantization failed: {str(e)}"

        if enable_onnx:
            status.update({"progress": 0.6, "current_step": "ONNX Conversion"})
            status_updates.append("Converting to ONNX format")

            try:
                output_dir = str(resource_manager.temp_dirs["onnx"])
                onnx_result = convert_to_onnx(model_name, task, output_dir)

                if onnx_result is None:
                    return {
                        "status": "Error",
                        "progress": 0.6,
                        "current_step": "ONNX Conversion Failed",
                    }, "ONNX conversion failed."

                if onnx_quantization != "None":
                    status_updates.append(f"Applying {onnx_quantization} ONNX quantization")
                    quantize_onnx_model(output_dir, onnx_quantization)

                status.update({"progress": 0.8, "current_step": "Pushing ONNX Model"})
                status_updates.append("Pushing ONNX model to Hub")
                result, push_message = push_to_hub(
                    local_path=output_dir,
                    repo_name=f"{repo_name}-onnx",
                    hf_token=hf_token,
                    tags=["onnx", "optimum", task],
                )
                status_updates.append(push_message)
            except Exception as e:
                logger.error(f"ONNX error: {str(e)}", exc_info=True)
                return {
                    "status": "Error",
                    "progress": 0.6,
                    "current_step": "ONNX Processing Failed",
                }, f"ONNX processing failed: {str(e)}"

        if quantization_type != "None" and quantized_model_path:
            status.update({"progress": 0.9, "current_step": "Pushing Quantized Model"})
            status_updates.append("Pushing quantized model to Hub")
            result, push_message = push_to_hub(
                local_path=quantized_model_path,
                repo_name=f"{repo_name}-quantized",
                hf_token=hf_token,
                tags=["quantized", task, quantization_type],
            )
            status_updates.append(push_message)

        status.update({"progress": 1.0, "status": "Complete", "current_step": "Completed"})
        cleanup_message = resource_manager.cleanup_temp_files()
        status_updates.append(cleanup_message)

        return status, "\n".join(status_updates)

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        return {
            "status": "Error",
            "progress": 0,
            "current_step": "Process Failed",
        }, f"An error occurred: {str(e)}"

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🤗 Model Conversion Hub
    Convert and optimize your Hugging Face models with quantization and ONNX support.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            model_name = gr.Textbox(label="Model Name", placeholder="e.g., bert-base-uncased")
            task = gr.Dropdown(choices=list(MODEL_CLASSES.keys()), label="Task", value="text_classification")

            with gr.Group():
                gr.Markdown("### Quantization Settings")
                quantization_type = gr.Dropdown(choices=["None", "4-bit", "8-bit", "16-bit-float"], label="Quantization Type", value="None")
                test_text = gr.Textbox(label="Test Text", placeholder="Enter text for model evaluation", lines=3, visible=False)

            with gr.Group():
                gr.Markdown("### ONNX Settings")
                enable_onnx = gr.Checkbox(label="Enable ONNX Conversion")
                with gr.Group(visible=False) as onnx_group:
                    onnx_quantization = gr.Dropdown(choices=["None", "8-bit", "16-bit-int", "16-bit-float"], label="ONNX Quantization", value="None")

            with gr.Group():
                gr.Markdown("### HuggingFace Settings")
                hf_token = gr.Textbox(label="HuggingFace Token (Required)", type="password")
                repo_name = gr.Textbox(label="Repository Name")

        with gr.Column(scale=1):
            status_output = gr.JSON(label="Status", value={"status": "Ready", "progress": 0, "current_step": "Waiting"})
            progress_bar = gr.Progress()
            message_output = gr.Markdown(label="Progress Messages")
            memory_info = gr.JSON(label="Resource Usage")

            convert_btn = gr.Button("🚀 Start Conversion", variant="primary")
            cleanup_btn = gr.Button("🧹 Cleanup Files", variant="secondary")

            with gr.Accordion("ℹ️ Help", open=False):
                gr.Markdown("""
                ### Quick Guide
                1. Enter your model name and HuggingFace token.
                2. Select the appropriate task.
                3. Choose optimization options.
                4. Click Start Conversion.

                ### Tips
                - Ensure sufficient system resources.
                - Use test text to validate conversions.
                - Clean up temporary files after completion.
                """)

    def update_memory_info():
        resource_manager = ResourceManager()
        return resource_manager.get_memory_info()

    quantization_type.change(lambda x: gr.update(visible=x != "None"), inputs=[quantization_type], outputs=[test_text])
    task.change(lambda x: gr.update(value=EXAMPLE_TEXTS.get(x, "")), inputs=[task], outputs=[test_text])
    enable_onnx.change(lambda x: gr.update(visible=x), inputs=[enable_onnx], outputs=[onnx_group])

    convert_btn.click(
        process_model,
        inputs=[model_name, task, quantization_type, enable_onnx, onnx_quantization, hf_token, repo_name, test_text],
        outputs=[status_output, message_output]
    ).then(update_memory_info, outputs=[memory_info])

    cleanup_btn.click(lambda: ResourceManager().cleanup_temp_files(), outputs=[message_output]).then(update_memory_info, outputs=[memory_info])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
