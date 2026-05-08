# cli.py
import torch
from lightning.pytorch.cli import LightningCLI

# Enables Tensor Core usage on Ampere+ (sm_80+) and Blackwell (sm_120) GPUs.
# "high" gives a good speed boost with negligible precision loss for probing.
torch.set_float32_matmul_precision("high")


def main():
    LightningCLI(
        save_config_kwargs={"overwrite": True},
        subclass_mode_model=True,
        subclass_mode_data=True,
    )


if __name__ == "__main__":
    main()
