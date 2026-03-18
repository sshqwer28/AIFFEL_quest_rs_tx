from typing import Optional

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
from peft import PeftModel

from ..base import Actor


class GPTActor_lora(Actor):
    """
    GPT Actor model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (GPT2Config): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): Rank of the LoRa layer.
        lora_train_bias (str): Bias training strategy for the LoRa layer.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 config: Optional[GPT2Config] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        if pretrained is not None:
            base_model = AutoModelForCausalLM.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5", quantization_config=bnb_config,device_map="auto")
            model = PeftModel.from_pretrained(base_model, pretrained)
        
        super().__init__(model, lora_rank, lora_train_bias)
