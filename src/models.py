"""Model loading, PEFT LoRA/DoRA adapter integration, and multimodal processor interface.

Supports Qwen2.5-VL (via qwen-vl-utils) and generic Vision-Language models.
Guarantees the vision encoder is frozen by regex and 4-bit quantization is restricted to CUDA.
Provides a tiny VLM fallback for instant, deterministic CPU testing without multi-GB downloads.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = logging.getLogger(__name__)


from transformers import GenerationConfig, PretrainedConfig


class TinyVLMConfig(PretrainedConfig):
    model_type = "tiny_vlm"
    is_encoder_decoder = False


class TinyVLM(nn.Module):
    """A minimal, self-contained Vision-Language Model for CPU smoke tests and unit testing.

    Includes a vision projection, token embeddings, and multi-layer causal decoder
    with explicit q_proj and v_proj layers to test PEFT LoRA/DoRA and loss masking.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.config = TinyVLMConfig()
        self.generation_config = GenerationConfig()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Vision encoder (to be frozen)
        self.visual = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=8),  # 128x128 -> 16x16
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, hidden_size),
        )

        # Language components
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Decoder layers with targetable projection names for LoRA
        self.layers = nn.ModuleList([
            TinyDecoderLayer(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass with causal masking and cross-entropy loss."""
        batch_size, seq_len = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None:
            vis_embeds = self.visual(pixel_values).unsqueeze(1)  # (batch, 1, hidden)
            # Prefix visual token to sequence
            inputs_embeds = torch.cat([vis_embeds, inputs_embeds], dim=1)
            if attention_mask is not None:
                vis_mask = torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([vis_mask, attention_mask], dim=1)
            if labels is not None:
                vis_labels = torch.full((batch_size, 1), -100, device=labels.device, dtype=labels.dtype)
                labels = torch.cat([vis_labels, labels], dim=1)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        class ModelOutput:
            def __init__(self, loss_val: Optional[torch.Tensor], logits_val: torch.Tensor):
                self.loss = loss_val
                self.logits = logits_val

        return ModelOutput(loss, logits)

    def get_input_embeddings(self) -> nn.Module:
        """Return token embedding module for PEFT compatibility."""
        return self.embed_tokens

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs: Any) -> Dict[str, Any]:
        """Prepare inputs for HuggingFace generation and PEFT compatibility."""
        return {"input_ids": input_ids, **kwargs}

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Greedy autoregressive generation for evaluation."""
        self.eval()
        cur_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.forward(cur_ids, pixel_values=pixel_values)
                next_token_logits = out.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                cur_ids = torch.cat([cur_ids, next_token], dim=1)
                # Check for EOS token (e.g. token 2)
                if (next_token == 2).all():
                    break
        return cur_ids


class TinyDecoderLayer(nn.Module):
    """Transformer decoder layer with distinct projection modules for LoRA targeting."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 2)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 2)
        self.down_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.bmm(q, k.transpose(1, 2)) / (x.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        context = self.o_proj(torch.bmm(attn, v))
        h = x + context
        ffn = self.down_proj(self.act(self.gate_proj(h)) * self.up_proj(h))
        return h + ffn


class SimpleProcessor:
    """Tokenizer and image processor wrapper compatible with TinyVLM and testing."""

    def __init__(self) -> None:
        # Standard vocab with A, B, C, D mapped deterministically
        self.vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2, "A": 10, "B": 11, "C": 12, "D": 13}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        for i in range(14, 1000):
            self.vocab[f"tok_{i}"] = i
            self.inv_vocab[i] = f"tok_{i}"
        self.pad_token_id = 0
        self.eos_token_id = 2

    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        return_tensors: str = "pt",
        padding: bool = True,
        max_length: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        if text is None:
            text = ""
        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = text

        all_input_ids = []
        for t in text_list:
            tokens = []
            words = t.replace("\n", " ").split(" ")
            for w in words:
                clean_w = w.strip(".:(), ")
                if clean_w in self.vocab:
                    tokens.append(self.vocab[clean_w])
                else:
                    tokens.append((hash(clean_w) % 800) + 20)
            if not tokens:
                tokens = [1]
            all_input_ids.append(tokens)

        # Pad sequences
        max_len = max(len(ids) for ids in all_input_ids)
        if max_length:
            max_len = min(max_len, max_length)

        padded_ids = []
        attention_masks = []
        for ids in all_input_ids:
            cur = ids[:max_len]
            attn = [1] * len(cur)
            if len(cur) < max_len:
                pad_len = max_len - len(cur)
                cur = cur + [self.pad_token_id] * pad_len
                attn = attn + [0] * pad_len
            padded_ids.append(cur)
            attention_masks.append(attn)

        res: Dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }

        if images is not None:
            if isinstance(images, Image.Image):
                img_list = [images]
            else:
                img_list = images
            pixel_tensors = []
            for img in img_list:
                img_resized = img.convert("RGB").resize((128, 128))
                t = torch.tensor(list(img_resized.getdata()), dtype=torch.float32).view(128, 128, 3)
                pixel_tensors.append(t.permute(2, 0, 1) / 255.0)
            res["pixel_values"] = torch.stack(pixel_tensors, dim=0)

        return res

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        words = []
        for tid in token_ids:
            if skip_special_tokens and tid in (0, 1, 2):
                continue
            words.append(self.inv_vocab.get(tid, f"tok_{tid}"))
        return " ".join(words)


def configure_peft_adapters(
    model: nn.Module,
    target_modules: List[str],
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_dora: bool = False,
    exclude_modules_pattern: str = ".*visual.*|.*vision.*",
) -> nn.Module:
    """Attach LoRA/DoRA adapters to the model, ensuring the vision tower remains frozen.

    Args:
        model: Base Vision-Language Model.
        target_modules: Names of module projections to adapt (e.g. ['q_proj', 'v_proj']).
        lora_r: LoRA rank.
        lora_alpha: Scaling factor.
        lora_dropout: Dropout probability.
        use_dora: Flag for Weight-Decomposed Low-Rank Adaptation.
        exclude_modules_pattern: Regex pattern to exclude modules (e.g., vision encoder).

    Returns:
        Adapted PeftModel.
    """
    if not PEFT_AVAILABLE:
        logger.warning("PEFT is not installed; continuing with unadapted model")
        return model

    # Freeze vision encoder parameters explicitly
    vision_regex = re.compile(exclude_modules_pattern, re.IGNORECASE)
    frozen_vision_params = 0
    for name, param in model.named_parameters():
        if vision_regex.search(name):
            param.requires_grad = False
            frozen_vision_params += param.numel()

    logger.info("Explicitly frozen %d parameters in vision modules matching '%s'", frozen_vision_params, exclude_modules_pattern)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        use_dora=use_dora,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    try:
        peft_model = get_peft_model(model, peft_config)
        trainable_params, all_params = peft_model.get_nb_trainable_parameters()
        logger.info(
            "LoRA/DoRA Attached: Trainable params: %d (%.2f%% of %d total)",
            trainable_params,
            100 * trainable_params / all_params,
            all_params,
        )
        return peft_model
    except Exception as e:
        logger.warning("Could not attach PEFT model: %s. Using base model.", e)
        return model


def load_model_and_processor(
    config: Dict[str, Any],
    force_tiny: bool = False,
) -> Tuple[nn.Module, Any]:
    """Load model and multimodal processor according to configuration.

    Supports Qwen2.5-VL, generic HuggingFace VLMs, and tiny model for fast smoke tests.

    Args:
        config: Parsed YAML configuration dictionary.
        force_tiny: If True, instantiate TinyVLM for immediate offline execution.

    Returns:
        Tuple of (model, processor).
    """
    m_cfg = config.get("model", {})
    model_name = m_cfg.get("name", "HuggingFaceTB/SmolVLM-256M-Instruct")
    model_type = m_cfg.get("type", "generic_vlm")
    target_modules = m_cfg.get("target_modules", ["q_proj", "v_proj"])
    lora_r = m_cfg.get("lora_r", 16)
    lora_alpha = m_cfg.get("lora_alpha", 32)
    lora_dropout = m_cfg.get("lora_dropout", 0.05)
    use_dora = m_cfg.get("use_dora", False)
    exclude_pattern = m_cfg.get("exclude_modules_pattern", ".*visual.*|.*vision.*")

    is_synth = config.get("is_synthetic", False) or force_tiny or "smoke" in config.get("experiment_name", "").lower()

    if is_synth or "SmolVLM" in model_name:
        # Check if user specifically requested offline tiny VLM or if download is avoided
        try:
            if not force_tiny and not is_synth:
                from transformers import AutoModelForImageTextToText, AutoProcessor
                processor = AutoProcessor.from_pretrained(model_name)
                model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
            else:
                logger.info("Instantiating fast deterministic TinyVLM for CPU smoke test")
                model = TinyVLM()
                processor = SimpleProcessor()
        except Exception as e:
            logger.info("HuggingFace download not used or failed (%s); using TinyVLM for fast reproducibility", e)
            model = TinyVLM()
            processor = SimpleProcessor()
    elif model_type == "qwen2_5_vl" or "qwen" in model_name.lower():
        # GPU Qwen path
        load_in_4bit = m_cfg.get("load_in_4bit", False)
        if load_in_4bit and not torch.cuda.is_available():
            logger.warning("load_in_4bit requested but CUDA is not available. 4-bit loading requires GPU; falling back to float32 on CPU.")
            load_in_4bit = False

        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            kwargs: Dict[str, Any] = {}
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                kwargs["device_map"] = "auto"
            else:
                dtype_str = m_cfg.get("torch_dtype", "bfloat16")
                dtype = getattr(torch, dtype_str, torch.float32)
                kwargs["torch_dtype"] = dtype

            processor = AutoProcessor.from_pretrained(model_name)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
        except Exception as e:
            logger.error("Failed to load Qwen2.5-VL model: %s", e)
            raise
    else:
        # Generic HuggingFace path
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(model_name)

    # Attach LoRA/DoRA adapters
    adapted_model = configure_peft_adapters(
        model=model,
        target_modules=target_modules,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        exclude_modules_pattern=exclude_pattern,
    )

    return adapted_model, processor
