# kuavo_train

`kuavo_train` provides an encapsulation based on **LeRobot Diffusion Policy**, supporting depth imaging, Transformer architecture as well as fusing multimodal information. Through this encapsulation, the end user can freely construct and train their own customised policies with their own tasks.

## Characteristics

- **Diffusion Policy encapsulation**: Encapsulates training, configuration as well as inferencing, in an easy-to-use package.
- **Transformer support**: Policy natively supports Transformer architecture, enhancing the ability to model temporal features.
- **Depth Image Processing and Fusion:**: It can process depth images from multiple cameras and supports feature fusion.
- **Highly Expendable**: Other strategies can be customized by referencing existing encapsulations and through inheritance or modification.

## File Structure

```
kuavo_train/
├── utils/
│ ├── __init__.py
│ ├── augmenter.py # Data Augmenter
│ ├── transforms.py # Data augmentation based on lerobot, with transformconfigs
│ └── utils.py # Other assistive utilities, such as saving and loading random number states
├── wrapper/
│ ├── dataset/
│ │ └── LeRobotDatasetWrapper.py # Dataset encapsulation, provides only the dataset inheritence examples, which is not yet used
│ ├── policy/
│ │ └── diffusion/
│ │ ├── __init__.py
│ │ ├── DiffusionConfigWrapper.py   # diffusion config inheritence example
│ │ ├── DiffusionModelWrapper.py    # diffusion model inheritence example, with depth imaging and feature fusion support
│ │ ├── DiffusionPolicyWrapper.py   # diffusion policy inheritence example, containing input processing such as crop and resize
│ │ ├── DiT_1D_model.py             # DiT-based 1D data diffusion model, optional
│ │ ├── DiT_model.py                # DiT-based diffusion model, optional
│ │ └── transformer_diffusion.py    # Transformer-based diffusion model
│ └── __init__.py
├── README.md
├── train_policy.py                 # Start here for policy training
└── train_policy_with_accelerate.py # Start here for policy training with Accelerate (multi-GPU)
```

## How to Use

1. **Prepare Dataset**  
   Use `python kuavo_data/CvtRosbag2Lerobot.py` to convert dataset into Lerobot parquets.

2. **Config Policy**  
   Use `configs/policy/diffusion_config.yaml` to set up training, model and policy parameters.

3. **Train Policy**  
   Execute `python kuavo_train/train_policy.py` to begin training.

4. **Extending Towards Other Policies**  
   - Please refer to `DiffusionPolicyWrapper.py` for inheritence structures, and customise your own policy.
   - If such policy requires depth image processing as well as fusing multimodal information, refer to `DiffusionModelWrapper.py`.
   - Diffusion-based models can use existing Transformer modules here.

## Dependencies

- PyTorch
- Torchvision
- Other dependencies listed in `requirements_ilcode.txt`, as well as those listed in [README.md](../README.md)

---


---

### Model Training

Use the existing preconverted lerobot dataset to start imitation learning based training:

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=diffusion_config.yaml \
  task=your_task_name \
  method=your_method_name \
  root=/path/to/lerobot_data/lerobot \
  training.batch_size=128 \
  policy_name=diffusion
```

Details:

* `task`: customized, task name (preferably corresponding to the task definition in the number transfer), such as `pick and place`
* `method`: custom, method name, used to distinguish different trainings, such as `diffusion_bs128_usedepth_nofuse`, etc.
* `root`: The local path of the training data. Note that lerobot is added. It needs to correspond to the data transfer saving path in 1, which is: `/path/to/lerobot_data/lerobot`
* `training.batch_size`: Batch size, can be adjusted according to GPU memory
* `policy_name`: The policy used, used for policy instantiation, currently supports `diffusion`, `act`, `lingbot`
* For other parameters, please refer to the yaml file description. It is recommended to modify the yaml file directly to avoid command line input errors.

---

### Model Training with multi-GPU Setup

Install accelerate: pip install accelerate

```bash
accelerate launch --config_file ./configs/policy/accelerate_config.yaml \ 
  ./kuavo_train/train_policy_with_accelerate.py  --  \ 
  --config-path ./configs/policy \ 
  --config-name diffusion_config.yaml
```

Details:

* diffusion_config.yaml config options are the same as above.

---

> This directory is mainly for robot imitation learning policy learning research and can be used as a template for quickly building diffusion-based policies.

---

### LingBot-VLA training integration

The warehouse has integrated `policy_name=lingbot` distribution logic, and LingBot's `torchrun` training process can be started directly from `kuavo_train/train_policy.py`.

Example:

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy \
  --config-name=lingbot_config.yaml \
  policy.dry_run=true
```

Related documents:

- `configs/policy/lingbot_config.yaml`
- `configs/policy/lingbot/robotwin_load20000h.yaml`
- `kuavo_train/train_lingbot.sh`
- `kuavo_train/wrapper/policy/lingbot/`
