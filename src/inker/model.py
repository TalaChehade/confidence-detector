import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"


def load_model_and_tokenizer(
    model_name=DEFAULT_MODEL_NAME,
    hf_token=None,
    load_in_4bit=True,
    compute_dtype=torch.float16,
    use_double_quant=True,
    padding_side="left",
    pad_token_id=0,
):
    """
    Load the Mistral tokenizer/model with the settings used in the original
    Colab implementation.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side=padding_side,
        token=hf_token,
    )
    tokenizer.pad_token_id = pad_token_id

    kwargs = {
        "device_map": "auto",
        "output_hidden_states": True,
        "token": hf_token,
    }

    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    return tokenizer, model
