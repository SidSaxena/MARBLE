<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="marble/utils/assets/marble-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="marble/utils/assets/marble-logo-light.svg">
    <!-- 这里的 img 用来做默认回退，路径可以指向 light 模式的那张图 -->
    <img src="marble/utils/assets/marble-logo-light.svg" alt="Marble Logo">
  </picture>
  <br/>
  <br/>
</p>


<h3 align="center">
    <p>State-of-the-art pretrained music models for training, evaluation, inference </p>
</h3>

Marble is a modular, configuration-driven suite for training, evaluating, and performing inference on state-of-the-art pretrained music models. It leverages LightningCLI to provide easy extensibility and reproducibility.

## News and Updates
* 📌 Join Us on MIREX Discord! [<img alt="join discord" src="https://img.shields.io/discord/1379757597984296980?color=%237289da&logo=discord"/>](https://discord.gg/YxP7VkNxjk)
* 2025-06-04 Now MARBLE v2 is available!


## Key Features
1. **Modularity**: Each component—encoders, tasks, transforms, decoders—is isolated behind a common interface. You can mix and match without touching core logic.
2. **Configurability**: All experiments are driven by YAML configs. No code changes are needed to switch datasets, encoders, or training settings.
3. **Reusability**: Common routines (data loading, training loop, metrics) are implemented once in `BaseTask`, `LightningDataModule`, and shared modules.
4. **Extensibility**: Adding new encoders or tasks requires implementing a small subclass and registering it via a config.

```text
┌──────────────────┐
│ DataModule       │  yields (waveform, label, path), optional audio transforms
└─▲────────────────┘
  │
  │ waveform                     Encoded →   hidden_states[B, L, T, H]
  ▼
┌─┴───────────────┐   embedding transforms (optional)
│ Encoder         │ ────────────────────────────────────────────────────┐
└─────────────────┘                                                     │
                                                                        ▼
                                                         (LayerSelector, TimeAvgPool…)
                                                                        │
                                                                        ▼
                                      ┌─────────────────────────────────┴──┐
                                      │ Decoder(s)                         │
                                      └────────────────────────────────────┘
                                                                  │ logits
                                                                  ▼
                                                   Loss ↔ Metrics ↔ Callbacks
```


## Getting Started

1. **Install dependencies**:
    ```bash
    # 1. create a new conda env
    conda create -n marble python=3.10 -y
    conda activate marble

    # 2. install ffmpeg
    conda install -c conda-forge ffmpeg -y

    # 3. now install other dependencies
    pip install -e .

    # 4. [Optional] downgrade pip to 24.0 if you are using fairseq modules
    # pip install pip==24.0
    # pip install fairseq
    ```

2. **Prepare data**: `python download.py all`.

3. **Configure**: Copy an existing YAML from `configs/` and edit paths, encoder settings, transforms, and task parameters.

4. **Run**:

   ```bash
   python cli.py fit --config configs/probe.MERT-v1-95M.GTZANGenre.yaml
   python cli.py test --config configs/probe.MERT-v1-95M.GTZANGenre.yaml
   ```

5. **Results**: Checkpoints and logs will be saved under `output/` and logged in Weights & Biases.







## Project Structure
```bash
.
├── marble/                   # Core code package
│   ├── core/                 # Base classes (BaseTask, BaseEncoder, BaseTransform)
│   ├── encoders/             # Wrapper classes for various SSL encoders
│   ├── modules/              # Shared transforms, callbacks, losses, decoders
│   ├── tasks/                # Downstream tasks (probe, few-shot, datamodules)
│   └── utils/                # IO utilities, instantiation helpers
├── cli.py                    # Entry-point for launching experiments
├── configs/                  # Experiment configs (YAML)
├── data/                     # Datasets and metadata files
├── scripts/                  # Run scripts & utilities
├── tests/                    # Unit tests for transforms & datasets
├── pyproject.toml            # Python project metadata
└── README.md                 # This file
```



## 🚀 Adding a New Encoder

Marble supports two flexible extension modes for encoders:

### Mode 1: **Internal Extension**
1. **Implement your encoder** under `marble/encoders/`:
   ```python
   # marble/encoders/my_encoder.py
   from marble.core.base_encoder import BaseAudioEncoder

   class MyEncoder(BaseAudioEncoder):
      def __init__(self, arg1, arg2):
         super().__init__()
         # initialize your model

      def forward(self, waveforms):
         # return List[Tensor] of shape (batch, layer, seq_len, hidden_size)
         # or return a dict of representations
   ```
2. **Reference it in your YAML**:

   ```yaml
   model:
     encoder:
       class_path: marble.encoders.my_encoder.MyEncoder
       init_args:
         arg1: 123
         arg2: 456
   ```

### Mode 2: **External Extension**

1. Place `my_encoder.py` anywhere in your project (e.g. `./my_project/my_encoder.py`).
2. Use the full import path in your YAML:

   ```yaml
   model:
     encoder:
       class_path: my_project.my_encoder.MyEncoder
       init_args:
         arg1: 123
   ```

> **Optional:**
>
> * If your encoder needs embedding-level transforms, implement a `BaseEmbTransform` subclass and register under `emb_transforms`.
> * If you need custom audio preprocessing, subclass `BaseAudioTransform` and register under `audio_transforms`.

  ```yaml
  emb_transforms:
    - class_path: marble.modules.transforms.MyEmbTransform
      init_args:
        param: value

  audio_transforms:
    train:
      - class_path: marble.modules.transforms.MyAudioTransform
        init_args:
          param: value
  ```


## 🚀 Adding a New Task

Marble supports two extension modes for tasks as well:

### Mode 1: **Internal Extension**

1. **Create a new task package** under `marble/tasks/YourTask/`:

   ```
   marble/tasks/YourTask/
   ├── __init__.py
   ├── datamodule.py    # Your LightningDataModule subclass
   └── probe.py          # Your BaseTask subclass, e.g. probe, finetune, fewshot
   ```

2. **Implement your classes**:

   ```python
   # datamodule.py
   import pytorch_lightning as pl

   class YourDataModule(pl.LightningDataModule):
       def setup(self, stage=None):
           ...
       def train_dataloader(self):
           ...
       # val_dataloader, test_dataloader, etc.

   # probe.py
   from marble.core.base_task import BaseTask

   class YourTask(BaseTask):
       def __init__(self, encoder, emb_transforms, decoders, losses, metrics, sample_rate, use_ema):
           super().__init__(...)
           # custom behavior here
   ```

3. **Point your YAML** to these classes:

   ```yaml
   task:
     class_path: marble.tasks.YourTask.probe.YourTask
     init_args:
       sample_rate: 22050
       use_ema: false

   data:
     class_path: marble.tasks.YourTask.datamodule.YourDataModule
   ```

### Mode 2: **External Extension**

1. Place your task code anywhere in your project (e.g. `./my_project/probe.py`, `./my_project/datamodule.py`).
2. Reference via full import path:

   ```yaml
   model:
     class_path: my_project.probe.CustomTask

   data:
     class_path: my_project.datamodule.CustomDataModule
   ```


# Citation
```text
@article{yuan2023marble,
  title={Marble: Music audio representation benchmark for universal evaluation},
  author={Yuan, Ruibin and Ma, Yinghao and Li, Yizhi and Zhang, Ge and Chen, Xingran and Yin, Hanzhi and Liu, Yiqi and Huang, Jiawen and Tian, Zeyue and Deng, Binyue and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={39626--39647},
  year={2023}
}
```
