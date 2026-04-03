# kuavo_train

`kuavo_train` provides a package based on **LeRobot Diffusion Policy**, supporting deep image processing, Transformer structure and multi-modal information fusion. Through this package, users can easily build and train their own policy models in customized tasks.

## Features

- **Diffusion Policy Encapsulation**: Encapsulates the training, configuration and inference processes for easy and direct use.
- **Transformer support**: Policy network supports Transformer structure to improve timing feature modeling capabilities.
- **Depth image processing and fusion**: Can process multi-camera depth images and support feature fusion.
- **Easy to Expand**: Other strategies can refer to existing packages and be customized through inheritance or modification.

## File structure

```
kuavo_train/
├── utils/
│ ├── __init__.py
│ ├── augmenter.py # Data augmentation tool
│ ├── transforms.py # Data augmentation based on lerobot, including transformconfigs
│ └── utils.py # Other auxiliary functions, such as saving and loading random number status
├── wrapper/
│ ├── dataset/
│ │ └── LeRobotDatasetWrapper.py # Dataset encapsulation, only provides examples of data set inheritance, and is not currently used.
│ ├── policy/
│ │ └── diffusion/
│ │ ├── __init__.py
│ │ ├── DiffusionConfigWrapper.py # diffusion config policy configuration inheritance example
│ │ ├── DiffusionModelWrapper.py # Diffusion model model inheritance example, including depth image, feature fusion, etc.
│ │ ├── DiffusionPolicyWrapper.py # Diffusion policy inheritance example, including cropping, scaling resize and other input processing
│ │ ├── DiT_1D_model.py # One-dimensional data diffusion model based on DiT magic modification, optional
│ │ ├── DiT_model.py # DiT-based diffusion model, optional
│ │ └── transformer_diffusion.py # Transformer-based diffusion model
│ └── __init__.py
├── README.md
├── train_policy.py # Policy training entrance
└── train_policy_with_accelerate.py # Policy training entrance (based on accelerate library, single-machine multi-card training)
```

## Instructions for use

1. **Prepare Data Set**
   Use `python kuavo_data/CvtRosbag2Lerobot.py` to convert the dataset.

2. **Configuration Strategy**
   Set training, model, and policy parameters in `configs/policy/diffusion_config.yaml`.

3. **Training Strategy**
   Start training via `python kuavo_train/train_policy.py`.

4. **Expand other strategies**
   - You can refer to the inheritance structure of `DiffusionPolicyWrapper.py` to implement custom policies.
   - If the strategy needs to process depth images or fuse multi-modal information, please refer to `DiffusionModelWrapper.py`
   - The diffusion model can reuse existing Transformer modules.

## Dependencies

- PyTorch
- torchvision
- For other dependencies, please refer to `requirements_ilcode.txt`, etc., and the overall project [README.md](../README.md)

---


---

### Model training

Use the converted data for imitation learning training:

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

Description:

* `task`: customized, task name (preferably corresponding to the task definition in the number transfer), such as `pick and place`
* `method`: custom, method name, used to distinguish different trainings, such as `diffusion_bs128_usedepth_nofuse`, etc.
* `root`: The local path of the training data. Note that lerobot is added. It needs to correspond to the data transfer saving path in 1, which is: `/path/to/lerobot_data/lerobot`
* `training.batch_size`: Batch size, can be adjusted according to GPU memory
* `policy_name`: The policy used, used for policy instantiation, currently supports `diffusion` and `act`
* For other parameters, please refer to the yaml file description. It is recommended to modify the yaml file directly to avoid command line input errors.

---

### Model training: single-machine multi-card mode

Install the accelerate library: pip install accelerate

```bash
accelerate launch --config_file ./configs/policy/accelerate_config.yaml \ 
  ./kuavo_train/train_policy_with_accelerate.py  --  \ 
  --config-path ./configs/policy \ 
  --config-name diffusion_config.yaml
```

Description:

* For the configuration parameter settings in the diffusion_config.yaml file, please refer to the above "Model Training: Parameter Description"

---

> This directory is mainly for robot imitation learning policy learning research and can be used as a template for quickly building diffusion-based policies.