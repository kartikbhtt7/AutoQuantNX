# AutoQuantnx Handler Support documentation

## Model Handlers and Supported Architectures

### Causal Language Modeling (`causal_lm_handler.py`)
**ORT Class**: `ORTModelForCausalLM`  
**Supported Architectures**:
- Bloom
- CodeGen
- Falcon
- GPT-2
- GPT BigCode
- GPT-J
- GPT-Neo
- GPT-NeoX
- LLaMA

### Masked Language Modeling (`embedding_model_handler.py`)
**ORT Class**: `ORTModelForFeatureExtraction`  
**Supported Architectures**:
- BERT
- RoBERTa
- DistilBERT
- ALBERT
- CamemBERT
- ConvBERT
- DeBERTa
- DeBERTa-v2
- ELECTRA
- MobileBERT
- XLM
- XLM-RoBERTa

### Question Answering (`question_answering_handler.py`)
**ORT Class**: `ORTModelForQuestionAnswering`  
**Supported Architectures**:
- BERT
- RoBERTa
- ALBERT
- DistilBERT
- DeBERTa
- DeBERTa-v2
- ELECTRA
- CamemBERT
- XLM
- XLM-RoBERTa
- MobileBERT
- ConvBERT
- BART

### Sequence Classification (`sequence_classification_handler.py`)
**ORT Class**: `ORTModelForSequenceClassification`  
**Supported Architectures**:
- BERT
- RoBERTa
- ALBERT
- DistilBERT
- DeBERTa
- DeBERTa-v2
- ELECTRA
- MobileBERT
- XLM
- XLM-RoBERTa
- CamemBERT
- ConvBERT
- BART
- MBART

### Token Classification (`token_classification_handler.py`)
**ORT Class**: `ORTModelForTokenClassification`  
**Supported Architectures**:
- BERT
- RoBERTa
- ALBERT
- DistilBERT
- DeBERTa
- DeBERTa-v2
- ELECTRA
- MobileBERT
- XLM
- XLM-RoBERTa
- CamemBERT
- ConvBERT
- GPT-2
- Bloom

### Multiple Choice (`multiple_choice_handler.py`)
**ORT Class**: `ORTModelForMultipleChoice`  
**Supported Architectures**:
- ALBERT
- BERT
- CamemBERT
- ConvBERT
- DeBERTa-v2
- DistilBERT
- ELECTRA
- MobileBERT
- RoBERTa
- XLM
- XLM-RoBERTa

### Speech-to-Text (`whisper_handler.py`)
**ORT Class**: `ORTModelForSpeechSeq2Seq`  
**Supported Architectures**:
- Whisper
- Speech2Text

### Image Classification (`image_classification_handler.py`)
**ORT Class**: `ORTModelForImageClassification`  
**Supported Architectures**:
- ViT
- BEiT
- ConvNeXt
- DeiT
- ResNet
- Swin
- MobileNet

### Seq2Seq Language Modeling (`seq2seq_lm_handler.py`)
**ORT Class**: `ORTModelForSeq2SeqLM`  
**Supported Architectures**:
- BART
- T5
- MT5
- Marian
- Pegasus
- LongT5
- M2M-100