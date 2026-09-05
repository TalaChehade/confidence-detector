import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def batched_string_to_hiddens(
    texts,
    tokenizer,
    model,
    layers,
    batch_size=32,
    rep_token=-1,
    max_length=512,
):
    """
    Extract the representative token hidden state for every requested layer.

    With the configured left-padding tokenizer and rep_token=-1, the last
    position is the last real token for every sequence.
    """
    out = {layer: [] for layer in layers}

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        outputs = model(**inputs, output_hidden_states=True)

        for layer in layers:
            hs = (
                outputs.hidden_states[layer][:, rep_token, :]
                .float()
                .cpu()
                .numpy()
            )
            out[layer].append(hs)

    return {
        layer: np.vstack(v)
        for layer, v in out.items()
    }
