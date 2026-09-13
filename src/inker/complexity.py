"""Adaptive-RAG external query-complexity evaluator.

This module implements the external-knowledge component E used by the
INKER IE-KRT replication.

The complexity classifier is:

    T5-Large
        ↓
    4-bit NF4 base model
        ↓
    LoRA adapter trained on Adaptive-RAG classifier data
        ↓
    A / B / C probabilities

Adaptive-RAG labels:

    A = no retrieval
    B = single-step retrieval
    C = multi-step retrieval

For the INKER replication, the discrete labels are converted into a
continuous complexity score using:

    E = 0*P(A) + 0.5*P(B) + 1*P(C)

      = 0.5*P(B) + P(C)

The A/B/C -> {0, 0.5, 1} mapping is a replication assumption. It should
not be presented as an exact formula released by the INKER authors.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from peft import PeftModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


DEFAULT_ADAPTIVE_RAG_BASE_MODEL = "t5-large"

LABELS = ("A", "B", "C")

LABEL_VALUES = {
    "A": 0.0,
    "B": 0.5,
    "C": 1.0,
}


@dataclass
class ComplexityResult:
    """Output produced by the Adaptive-RAG complexity evaluator."""

    predicted_class: str

    p_A: float
    p_B: float
    p_C: float

    E: float

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "predicted_class": self.predicted_class,
            "p_A": self.p_A,
            "p_B": self.p_B,
            "p_C": self.p_C,
            "E": self.E,
        }


def _find_adapter_directory(root: Path) -> Path:
    """Find the directory containing adapter_config.json."""

    direct_config = root / "adapter_config.json"

    if direct_config.exists():
        return root

    matches = list(root.rglob("adapter_config.json"))

    if len(matches) == 1:
        return matches[0].parent

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No adapter_config.json was found inside {root}"
        )

    raise RuntimeError(
        "Multiple LoRA adapters were found inside "
        f"{root}. Please provide the exact adapter directory."
    )


def _prepare_adapter_path(adapter_path: str) -> Path:
    """Resolve an adapter directory or extract an adapter ZIP.

    This allows the config to point directly to the ZIP stored on
    Google Drive.

    Example:

        /content/drive/MyDrive/INKER_Models/
        adaptive_rag_t5_large_lora.zip
    """

    expanded = os.path.expandvars(
        os.path.expanduser(adapter_path)
    )

    path = Path(expanded)

    if not path.exists():
        raise FileNotFoundError(
            "Adaptive-RAG LoRA adapter was not found:\n"
            f"{path}\n\n"
            "If the model is stored in Google Drive, make sure "
            "Drive is mounted before running the experiment."
        )

    if path.is_dir():
        return _find_adapter_directory(path)

    if path.suffix.lower() != ".zip":
        raise ValueError(
            "adapter_path must point either to an extracted "
            "LoRA adapter directory or to a .zip archive."
        )

    # Extract into local Colab storage rather than repeatedly reading
    # model files from mounted Google Drive.
    extraction_dir = Path(
        "/content/inker_adaptive_rag_adapter"
    )

    marker = extraction_dir / ".extracted"

    if not marker.exists():
        if extraction_dir.exists():
            shutil.rmtree(extraction_dir)

        extraction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "Extracting Adaptive-RAG LoRA adapter..."
        )

        with zipfile.ZipFile(path, "r") as archive:
            archive.extractall(extraction_dir)

        marker.touch()

        print(
            "✓ Adapter extracted to:",
            extraction_dir,
        )

    return _find_adapter_directory(
        extraction_dir
    )


def _dtype_from_name(name: str) -> torch.dtype:
    """Resolve the requested 4-bit computation dtype."""

    name = str(name).lower()

    if name == "auto":
        if (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
            return torch.bfloat16

        return torch.float16

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    if name not in mapping:
        raise ValueError(
            f"Unsupported compute dtype: {name}"
        )

    return mapping[name]


class AdaptiveRAGComplexityEvaluator:
    """Adaptive-RAG T5-Large + LoRA complexity evaluator."""

    def __init__(
        self,
        adapter_path: str,
        base_model_name: str = DEFAULT_ADAPTIVE_RAG_BASE_MODEL,
        max_input_length: int = 384,
        load_in_4bit: bool = True,
        compute_dtype: str = "auto",
    ):
        if load_in_4bit and not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit BitsAndBytes complexity inference requires CUDA. "
                "Either run on a GPU runtime or set complexity.load_in_4bit=false."
            )

        self.base_model_name = base_model_name
        self.max_input_length = int(
            max_input_length
        )

        self.adapter_path = (
            _prepare_adapter_path(
                adapter_path
            )
        )

        print(
            "\nLoading Adaptive-RAG complexity evaluator"
        )

        print(
            "Base model:",
            self.base_model_name,
        )

        print(
            "LoRA adapter:",
            self.adapter_path,
        )

        # The classifier was trained on raw question text.
        # Therefore we deliberately DO NOT prepend:
        #
        # "Classify the complexity of the following query:"
        #
        # here.
        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.base_model_name
            )
        )

        if load_in_4bit:
            dtype = _dtype_from_name(
                compute_dtype
            )

            quantization_config = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                )
            )

            base_model = (
                AutoModelForSeq2SeqLM.from_pretrained(
                    self.base_model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            )

        else:
            base_model = (
                AutoModelForSeq2SeqLM.from_pretrained(
                    self.base_model_name,
                    device_map="auto",
                )
            )

        self.model = PeftModel.from_pretrained(
            base_model,
            str(self.adapter_path),
            is_trainable=False,
        )

        self.model.eval()

        print(
            "✓ Adaptive-RAG LoRA classifier loaded."
        )

    @property
    def device(self):
        """Return the device that should receive tokenizer inputs."""

        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return next(
                self.model.parameters()
            ).device

    @torch.no_grad()
    def _class_log_scores(
        self,
        question: str,
    ) -> torch.Tensor:
        """Compute sequence log-likelihood for A, B and C.

        We score the complete output sequence for each class instead
        of assuming that A/B/C correspond to a particular single
        tokenizer ID.
        """

        question = str(question).strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        inputs = self.tokenizer(
            question,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
        )

        labels = self.tokenizer(
            text_target=list(LABELS),
            return_tensors="pt",
            padding=True,
        )["input_ids"]

        batch_size = len(LABELS)

        input_ids = inputs[
            "input_ids"
        ].repeat(
            batch_size,
            1,
        )

        attention_mask = inputs[
            "attention_mask"
        ].repeat(
            batch_size,
            1,
        )

        input_ids = input_ids.to(
            self.device
        )

        attention_mask = attention_mask.to(
            self.device
        )

        labels = labels.to(
            self.device
        )

        decoder_input_ids = (
            self.model.prepare_decoder_input_ids_from_labels(
                labels=labels
            )
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )

        logits = outputs.logits

        log_probs = torch.log_softmax(
            logits,
            dim=-1,
        )

        safe_labels = labels.clone()

        mask = (
            safe_labels
            != self.tokenizer.pad_token_id
        )

        safe_labels[
            ~mask
        ] = 0

        token_log_probs = (
            log_probs.gather(
                dim=-1,
                index=safe_labels.unsqueeze(-1),
            )
            .squeeze(-1)
        )

        token_log_probs = (
            token_log_probs
            * mask
        )

        sequence_scores = (
            token_log_probs.sum(
                dim=-1
            )
        )

        return sequence_scores

    @torch.no_grad()
    def evaluate(
        self,
        question: str,
    ) -> ComplexityResult:
        """Evaluate one question."""

        scores = self._class_log_scores(
            question
        )

        probabilities = torch.softmax(
            scores.float(),
            dim=0,
        )

        p_A = float(
            probabilities[0].item()
        )

        p_B = float(
            probabilities[1].item()
        )

        p_C = float(
            probabilities[2].item()
        )

        predicted_index = int(
            torch.argmax(
                probabilities
            ).item()
        )

        predicted_class = (
            LABELS[predicted_index]
        )

        E = (
            LABEL_VALUES["A"] * p_A
            + LABEL_VALUES["B"] * p_B
            + LABEL_VALUES["C"] * p_C
        )

        return ComplexityResult(
            predicted_class=predicted_class,
            p_A=p_A,
            p_B=p_B,
            p_C=p_C,
            E=float(E),
        )

    def __call__(
        self,
        question: str,
    ) -> ComplexityResult:
        return self.evaluate(
            question
        )

    @classmethod
    def from_config(
        cls,
        config: dict,
        adapter_path: Optional[str] = None,
        base_model_name: Optional[str] = None,
    ):
        """Build evaluator directly from default.yaml."""

        complexity_config = config.get(
            "complexity",
            {},
        )

        resolved_adapter = (
            adapter_path
            or complexity_config.get(
                "adapter_path"
            )
        )

        if not resolved_adapter:
            raise ValueError(
                "No complexity.adapter_path was provided."
            )

        resolved_base_model = (
            base_model_name
            or complexity_config.get(
                "base_model_name",
                DEFAULT_ADAPTIVE_RAG_BASE_MODEL,
            )
        )

        return cls(
            adapter_path=resolved_adapter,
            base_model_name=resolved_base_model,
            max_input_length=complexity_config.get(
                "max_input_length",
                384,
            ),
            load_in_4bit=complexity_config.get(
                "load_in_4bit",
                True,
            ),
            compute_dtype=complexity_config.get(
                "compute_dtype",
                "auto",
            ),
        )
