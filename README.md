# 🤗 AutoQuantNX

## Overview

This repository contains a Gradio-based web application for converting and optimizing Hugging Face models. It supports:

1. Quantization of models using various methods (e.g., 4-bit, 8-bit, 16-bit-float).
2. Conversion of models to ONNX format for deployment and inference optimization.
3. Pushing converted/optimized models to the Hugging Face Hub.
4. Testing and comparing model performance before and after optimization.

## Features

### Supported Tasks
- Text Classification
- Named Entity Recognition (NER)
- Question Answering
- Causal Language Modeling
- Whisper (Speech-to-Text)
- Embedding Fine-Tuning

### Quantization Options
- None (default)
- 4-bit
- 8-bit
- 16-bit-float

### ONNX Conversion
- Converts models to ONNX format.
- Supports optional ONNX quantization (8-bit, 16-bit-int, 16-bit-float).

## File Structure

```
.
├── src
│   ├── gradio_app.py        # Gradio web application
│   ├── model_handlers.py    # Handlers for specific model tasks
│   ├── onnx_conversion.py   # ONNX conversion and quantization 
logic
│   ├── quantize.py          # Quantization management and performance testing
│   └── utilities.py         # Resource management and utility docs
└── README.md                # Documentation
```

## Prerequisites

### Using requirements.txt
1. Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Using Poetry
1. Install Poetry (if not already installed):

   if linux:
      ```bash
      Use curl -sSL https://install.python-poetry.org | python3 -
      ```
   else:

      Follow the official instructions to install Poetry on your system:
      ```bash
      https://python-poetry.org/docs/#installation
      ```

2. Install Dependencies: Navigate to the project directory and run the following command to install all the required dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the Virtual Environment: Poetry automatically creates a virtual environment for the project. To activate it, run:
   ```bash
   poetry shell
   ```

## Usage

### Launch the App

Run the following command to start the Gradio web application:
```bash
python src/gradio_app.py
```

The app will be accessible at `http://localhost:7860` by default.

### Steps to Use the App

1. **Enter Model Details**: Provide the Hugging Face model name and task.
2. **Select Optimization Options**: Configure quantization and/or ONNX conversion settings.
3. **Provide Hugging Face Token**: Enter your token for accessing and pushing models to the Hugging Face Hub.
4. **Start Conversion**: Click the "Start Conversion" button to process the model.
5. **Monitor Progress**: View status updates, resource usage, and results directly in the app.
6. **Push to Hub**: Optimized models are automatically pushed to your specified Hugging Face repository.
7. **Cleanup**: Use the "Cleanup Files" button to remove temporary files.

### Example

For a model `bert-base-uncased` performing text classification:
1. Select `text_classification` as the task.
2. Enable quantization (e.g., `8-bit`).
3. Enable ONNX conversion with optimization.
4. Click "Start Conversion" and monitor progress.

## Key Functions

### gradio_app.py
- **process_model**: Main function handling model quantization, ONNX conversion, and Hugging Face Hub integration.
- **update_memory_info**: Monitors and displays system resource usage.

### onnx_conversion.py
- **convert_to_onnx**: Converts models to ONNX format.
- **quantize_onnx_model**: Quantizes ONNX models for optimized inference.

### utilities.py
- **ResourceManager**: Manages temporary files and memory usage.
- **push_to_hub**: Pushes models to the Hugging Face Hub.

## Notes

- Ensure you have sufficient system resources for model conversion and quantization.
- Use the Hugging Face Hub token with proper write permissions for pushing models.
- Clean up temporary files regularly to free up disk space.

## Troubleshooting

1. **Model Conversion Fails**: Ensure the model and task are supported.
2. **Insufficient Resources**: Free up memory or reduce optimization levels.
3. **ONNX Quantization Errors**: Verify that the selected quantization type is supported for the model.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributions

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Optimum Library](https://github.com/huggingface/optimum)
- [Gradio](https://gradio.app/)
