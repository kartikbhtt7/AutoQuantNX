# 🤗 AutoQuantNX (**Still under development and testing phase**)

## Overview
AutoQuantNX is a powerful Gradio-based web application designed to simplify the process of optimizing and deploying Hugging Face models. It supports a wide range of tasks, including quantization, ONNX conversion, and seamless integration with the Hugging Face Hub. With AutoQuantNX, you can easily convert models to ONNX format, apply quantization techniques, and push the optimized models to your Hugging Face account—all through an intuitive user interface.

## Features

### Supported Tasks
AutoQuantNX supports the following tasks:

* Text Classification
* Named Entity Recognition (NER)
* Question Answering
* Causal Language Modeling
* Masked Language Modeling
* Sequence-to-Sequence Language Modeling
* Multiple Choice
* Whisper (Speech-to-Text)
* Embedding Fine-Tuning
* Image Classification (Placeholder for future implementation)

### Quantization Options
* None (default)
* 4-bit
* 8-bit
* 16-bit-float

### ONNX Conversion
Converts models to ONNX format for optimized deployment.

Supports optional ONNX quantization:
* 8-bit
* 16-bit-int
* 16-bit-float

### Hugging Face Hub Integration
* Automatically pushes optimized models to your Hugging Face Hub repository
* Tags models with metadata for easy identification (e.g., onnx, quantized, task type)

### Performance Testing
Compares original and quantized models using metrics like:
* Mean Squared Error (MSE)
* Spearman Correlation
* Cosine Similarity
* Inference Time
* Model Size

## File Structure
```
.
├── src
│   ├── app.py                                     # Gradio web application
│   ├── handlers/                                  # Task-specific model handlers
│   │   ├── base_handler.py                           # Base class for all handlers
│   │   ├── causal_lm_handler.py                      # Handler for causal language models
│   │   ├── embedding_model_handler.py                # Handler for embedding models
│   │   ├── question_answering_handler.py             # Handler for question answering
│   │   ├── sequence_classification_handler.py        # Handler for text classification
│   │   ├── token_classification_handler.py           # Handler for NER
│   │   ├── whisper_handler.py                        # Handler for Whisper models
│   │   ├── masked_lm_handler.py                      # Handler for masked language models
│   │   ├── seq2seq_lm_handler.py                     # Handler for sequence-to-sequence models
│   │   ├── multiple_choice_handler.py                # Handler for multiple-choice models
│   │   └── image_classification_handler.py           # Placeholder for image classification
│   ├── optimization/                              # Optimization logic
│   │   ├── onnx_conversion.py                        # ONNX conversion and quantization
│   │   └── quantize.py                               # Quantization management
│   ├── utilities/                                 # Utility functions
│   │   ├── push_to_hub.py                            # Pushing models to Hugging Face Hub
│   │   └── resources.py                              # Resource management
│   └── __init__.py                                # Package initialization
├── README.md                                      # Documentation
└── requirements.txt                               # Python dependencies
```

## Prerequisites

### Using requirements.txt
* Python 3.8 or higher
* Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Using Poetry
1. Install Poetry (if not already installed):
   
   Linux:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   Other platforms: Follow the official instructions.

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Usage

### Launch the App
Run the following command to start the Gradio web application:
```bash
python src/app.py
```
The app will be accessible at http://localhost:7860 by default.

### Steps to Use the App
1. Enter Model Details:
   * Provide the Hugging Face model name
   * Select the task type (e.g., text classification, question answering)

2. Select Optimization Options:
   * Choose quantization type (e.g., 4-bit, 8-bit)
   * Enable ONNX conversion and select quantization options if needed

3. Provide Hugging Face Token:
   * Enter your Hugging Face token for accessing and pushing models to the Hub

4. Start Conversion:
   * Click the "Start Conversion" button to process the model

5. Monitor Progress:
   * View real-time status updates, resource usage, and results directly in the app

6. Push to Hub:
   * Optimized models are automatically pushed to your specified Hugging Face repository

7. Cleanup:
   * Use the "Cleanup Files" button to remove temporary files

### Example
For a model like bert-base-uncased performing text classification:
1. Select text_classification as the task
2. Enable quantization (e.g., 8-bit)
3. Enable ONNX conversion with optimization
4. Click "Start Conversion" and monitor progress

## Key Functions

### app.py
* `process_model`: Main function handling model quantization, ONNX conversion, and Hugging Face Hub integration
* `update_memory_info`: Monitors and displays system resource usage

### optimization/onnx_conversion.py
* `convert_to_onnx`: Converts models to ONNX format
* `quantize_onnx_model`: Quantizes ONNX models for optimized inference

### optimization/quantize.py
* `ModelQuantizer`: Handles quantization of PyTorch models and performance testing

### utilities/push_to_hub.py
* `push_to_hub`: Pushes models to the Hugging Face Hub

### utilities/resources.py
* `ResourceManager`: Manages temporary files and memory usage

## Notes
* Ensure you have sufficient system resources for model conversion and quantization
* Use a Hugging Face Hub token with proper write permissions for pushing models
* Clean up temporary files regularly to free up disk space

## Troubleshooting
* Model Conversion Fails: Ensure the model and task are supported
* Insufficient Resources: Free up memory or reduce optimization levels
* ONNX Quantization Errors: Verify that the selected quantization type is supported for the model

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributions
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgments
* Hugging Face Transformers
* Optimum Library
* Gradio
* ONNX Runtime