import torch
import transformers
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- Quantization setup for 4-bit loading ---
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# --- Model & tokenizer setup ---
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    dtype=torch.float16,
    device_map="auto"
)

# --- EOS tokens ---
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# --- Chat function ---
def chat_func(message, history):
    messages = []

    # Reconstruct chat history properly
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    # Add the latest user message
    messages.append({"role": "user", "content": message})

    # Tokenize with chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate model output
    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    # Decode new tokens only
    response_ids = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    # Stream the output for Gradio
    partial_message = ""
    for char in response:
        partial_message += char
        yield partial_message


# --- Gradio UI ---
demo = gr.ChatInterface(
    fn=chat_func,
    title="Local Llama 3 Chatbot",
    description="Running Meta Llama 3 8B in 4-bit quantization locally!"
)

demo.launch()
