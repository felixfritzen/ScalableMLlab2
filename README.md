# Lab 2: Scalable Machine Learning

**Implementation:**
- **Training Notebook:** `FelixValter_Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning.ipynb`
- **UI Script:** `app.py`

---

## Task 1: Fine-Tune a Pre-Trained Large Language Model and Build a Serverless UI

### First Steps
1. Create a free account on [Hugging Face](https://huggingface.co).
2. Create a free account on [Google Colab](https://colab.google.com).

### Objectives
1. Fine-tune an existing pre-trained large language model on the FineTome Instruction Dataset.
2. Build and deploy an inference pipeline with a Gradio UI on Hugging Face Spaces.

**Relevant Links:**
- Dataset and code: [GitHub Repository](https://github.com/felixfritzen/ScalableMLlab2)
- Deployed UI: [Hugging Face Space](https://huggingface.co/spaces/felixfritzen/ScalableMLlab2)

---

## Task 2: Improve Model Performance and Fine-Tune Multiple LLMs

### README.md Requirements
Describe ways to improve model performance using:
- **Model-Centric Approaches:** 
  - Tune hyperparameters (e.g., rank for LoRA adapters, learning rate, decay).
  - Use a LoRA adapter for fine-tuning due to efficiency. For example:
    - Fully connected layer: 1M parameters (1000x1000 matrix).
    - LoRA adapter (rank 8): 16K parameters, reducing computation by 500x.

  **Approach:**
  - Perform a hyperparameter grid search for rank values (e.g., 8, 16, 32, 64).
  - Use random search for learning rate and decay.

- **Data-Centric Approaches:**
  - Use a larger dataset for training. The given dataset is a subset of a larger dataset.
  - Prioritize high-quality data to avoid degradation in performance.

**Base Model:**
Llama-3.2-3B-Instruct was chosen for final deployment based on evaluation loss.

---

### Fine-Tuning Details
1. Split the dataset (80% training, 20% validation).
2. Monitor validation loss during training.
3. Use the parameters with the best validation performance for the final model.
4. Limit training to 500 steps per rank due to time constraints.

**Validation Results:**
- Best hyperparameter setting: Rank `r=32`, with the lowest evaluation loss after 500 steps.

---

### Exploring Open-Source Foundation LLMs
1. Experimented with different open-source foundation models.
2. To mitigate CPU inference limitations, used GPU resources on Hugging Face for optimal performance.

---

### Fine-Tuning Framework
- Framework used: **Unsloth** for fine-tuning.

---

## Deliverables
1. Source code: [GitHub Repository](https://github.com/felixfritzen/ScalableMLlab2).
2. README.md file with performance improvement descriptions.
3. Publicly accessible UI: [Hugging Face Space](https://huggingface.co/spaces/felixfritzen/ScalableMLlab2).
