# LingBot Training Instructions

This article explains how to start LingBot training in the current warehouse, and the actual relationship between the two layers of configuration files.

## 1. Training entrance

Current training entry command:

```bash
python kuavo_train/train_policy.py --config-path=../configs/policy/ --config-name=lingbot_config.yaml
```

This command will go to the `policy_name=lingbot` branch, call `torchrun` from `kuavo_train/train_policy.py`, and finally execute:

```text
kuavo_train/lingbot/tasks/vla/train_lingbotvla.py
```

## 2. Two-tier configuration file

Training actually consists of two layers of configuration:

### Outer startup configuration

File:

```text
configs/policy/lingbot_config.yaml
```

Function:

- Specify the training entrance to use `lingbot`
- Specify the data root directory `root`
- Specify the model path `policy.model_path`
- Specify tokenizer path `policy.tokenizer_path`
- Specify the graphics card `training.gpu_ids` / `policy.env.CUDA_VISIBLE_DEVICES`
- Specify the outer output directory template `training.output_directory`
- Specify whether to continue training `training.resume`

### Inner LingBot training configuration

File:

```text
configs/policy/lingbot/robotwin_load20000h.yaml
```

Function:

- Control LingBot training details
- Includes default values for `train.*`, `data.*`, `model.*`

## 3. Configure priority

The priority is not a simple choice between two, but:

1. First read `lingbot_config.yaml`
2. Read `robotwin_load20000h.yaml` again
3. The outer launcher converts some parameters into command line parameters
4. Command line parameters cover the corresponding fields in `robotwin_load20000h.yaml`

That is to say:

- **No parameters passed by the outer layer**, finally use `robotwin_load20000h.yaml`
- **Parameters explicitly passed by the outer layer**, whichever is the outer layer

The key parameters currently covered by the outer layer include:

- `data.train_path`
- `train.action_dim`
- `train.micro_batch_size`
- `train.global_batch_size`
- `train.output_dir`
- `model.model_path`
- `model.tokenizer_path`
- Optional `model.moge_path`
- Optional `model.morgbd_path`

So if you change many `training.*` fields in `lingbot_config.yaml`, but the launcher does not transparently pass it to LingBot, then these fields will not automatically overwrite the inner training configuration.

## 4. Current recommended training methods

### 4.1 Check the critical path first

Please first confirm that these paths exist:

```text
/home/yunxi/lmy/VLA/lingbot-vla
/home/yunxi/lmy/VLA/lingbot-vla-4b
/home/yunxi/lmy/VLA/Qwen2.5-VL-3B-Instruct
/home/yunxi/lmy/VLA/huggingface/lerobot/pick_and_place0_4_2
```

And the current warehouse comes with:

```text
third_party/lerobot/src
```

The current training link has given priority to the following in this warehouse:

```text
third_party/lerobot/src
```

Instead of the history directory of the old warehouse.

### 4.2 Start training directly

```bash
python kuavo_train/train_policy.py --config-path=../configs/policy/ --config-name=lingbot_config.yaml
```

### 4.3 Just look at what command will eventually be started

If you just want to check the final generated `torchrun` command without actually training:

```bash
python kuavo_train/train_policy.py --config-path=../configs/policy/ --config-name=lingbot_config.yaml policy.dry_run=true
```

## 5. Current configuration recommendations

### Graphics card

If you want to train multiple cards, it is recommended to set it directly in `configs/policy/lingbot_config.yaml`:

```yaml
policy:
  env:
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
```

If `policy.env.CUDA_VISIBLE_DEVICES` is not set, the launcher will try to refer to `training.gpu_ids`.

### batch size

In the current outer configuration:

```yaml
training:
  batch_size: 1
```

will be converted to:

- `--train.micro_batch_size 1`
- `--train.global_batch_size = micro_batch_size * Number of GPUs * Number of nodes `

For example, when there are 8 cards, it will become:

```text
micro_batch_size=1
global_batch_size=8
```

If OOM occurs during single-card training, it is preferable to keep `batch_size: 1` and use multi-card sharding instead.

## 6. Continue training

In the outer configuration:

```yaml
training:
  resume: true
  resume_timestamp: "20260320_063225"
```

When `resume=true` and `resume_timestamp` is not empty, the launcher will switch the output directory back to the old run directory.

If you do not continue training:

```yaml
training:
  resume: false
```

will generate a new one:

```text
outputs/train/${task}/${method}/run_${timestamp}
```

## 7. How to modify training logic

### Want to change training details

Please modify it first:

```text
configs/policy/lingbot/robotwin_load20000h.yaml
```

For example:

- learning rate
- epoch
- save_steps
- FSDP / offload
- tokenizer length
- Other `train.*` / `data.*` default behaviors

### Want to change the behavior of the entry layer

Please modify:

```text
configs/policy/lingbot_config.yaml
```

For example:

- Data root directory
- model path
- tokenizer path
- Graphics card
- Whether to dry run
- whether to resume
- Output directory template

## 8. FAQ

### Training startup times `No module named kuavo_train.wrapper`

It means that the module path of the entry script is not processed properly. This problem has been fixed in the current warehouse. You can run it directly according to the command in this article.

### pydantic’s `UnsupportedFieldAttributeWarning` appears in the training log

This is usually just warning noise and not a cause of training failure. As long as training continues entering:

- `torchrun`
- `Prepare model`
- `Prepare data`
- `Start training`

You can ignore it first.

### Not enough video memory on a single card

LingBot-VLA is relatively large, and a single card is prone to OOM in the first optimizer step. A more stable approach is:

- `batch_size: 1`
- Use multiple cards
- Prioritize making `CUDA_VISIBLE_DEVICES` consistent with actual `nproc-per-node`

## 9. Recommended usage habits

- Put the real LingBot training parameters in `robotwin_load20000h.yaml`
- Put the outer path, graphics card, resume, and output directory in `lingbot_config.yaml`
- After modification, run `policy.dry_run=true`
- Confirm that the generated `torchrun` command is correct before starting formal training
