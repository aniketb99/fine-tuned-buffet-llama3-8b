import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load tokenizer and model (with LoRA)
@st.cache_resource
def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/meta-llama-3.1-8b-unsloth-bnb-4bit",
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    model = PeftModel.from_pretrained(base_model, "aniketb99/buffet_lora_llama3.1_7b_model")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/meta-llama-3.1-8b-unsloth-bnb-4bit")
    return model, tokenizer

model, tokenizer = load_model()

# Alpaca-style prompt
def build_prompt(instruction, question):
    alpaca_prompt = """Below is a financial question followed by an appropriate response from Warren Buffett’s perspective.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    return alpaca_prompt.format(instruction, f"Question: {question}", "")

# Streamlit UI
st.title("💬 BuffettBot: Ask Warren Buffett-style Financial Questions")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question (e.g., Should I invest in AI startups?)", key="input")

if st.button("Ask"):
    if user_input:
        st.session_state.chat_history.append(("user", user_input))

        instruction = "Answer the question like Warren Buffett would."
        prompt = build_prompt(instruction, user_input)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Response:" in decoded:
            response = decoded.split("### Response:")[-1].strip()
        else:
            response = decoded.strip()

        st.session_state.chat_history.append(("assistant", response))

# Display chat history
for speaker, msg in st.session_state.chat_history:
    if speaker == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**BuffettBot:** {msg}")
