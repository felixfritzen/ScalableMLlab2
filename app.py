from huggingface_hub import InferenceClient
from transformers import TextStreamer
from transformers import AutoTokenizer
import gradio as gr
import re
"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
#client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "felixfritzen/lora_model_final",
    max_seq_length =  2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

def get_assistant_text(output):
    assistant_start = "<|start_header_id|>assistant<|end_header_id|>"
    eot_marker = "<|eot_id|>"
    if assistant_start in output:
        start_index = output.index(assistant_start) + len(assistant_start)
        end_index = output.find(eot_marker, start_index)
        return output[start_index:end_index].strip()
    return 'Failed'

def apply_llm(prompt):
    message=[{"role": "user", "content":  str(prompt)}]
    inputs = tokenizer.apply_chat_template(
    message,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
    ).to("cuda")
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    outputs = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 64,
                  use_cache = True, temperature = 1.5, min_p = 0.1)
    response = tokenizer.decode(outputs[0])
    return str(get_assistant_text(response))


interface = gr.Interface(fn=apply_llm, inputs="text", outputs="text")

if __name__ == "__main__":
    interface.launch()
