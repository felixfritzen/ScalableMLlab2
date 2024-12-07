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
  We are using a LoRA adapter to finetune the model because it is cheaper than a fc layer. A fc layer would be represented by a 1000x1000 matrix therefore = 1M parameters, however a LoRA adapter of rank 8 would be represented by two matrices 1000x8 and 8x1000 therefore 16K parameters and 500 times less than fc layer. Note that the product is still [1000x8] @[8x1000]=[1000,1000]. Therefore the most impactful hyperparameter is this bottleneck, the rank which is 8 in  this case. Therefore we will do a hyperparameter search of the rank in powers of 2, e.g 8,16,32,64. We will do a grid here, we could also do a search of the learning rate and decay, here we would do a random search and not a grid. However due to the limited scope we choose to just do the hyperparameter search of the rank.

  **Chosen approach:**
  - Perform a hyperparameter grid search for rank values (e.g., 8, 16, 32, 64).

- **Data-Centric Approaches:**
  More data can have a large impact, the dataset given in the task is a subset of a 10 times larger dataset that could be used. However, more data means  more computation and more time. However quality is also important, using large amount of low quality data can be worse than low amount of good data. We could have some function that looks at the data before training and removes low quality.

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
