from dataclasses import dataclass

@dataclass
class MAEConfig:
    ckpt_path: str = "/home/wph52/causal-earth/causal_earth/ckpts/fmow_pretrain.pth"