## WandB Setup

In `main.py`, please replace `ooooo`(project name), `???`(experiment name), and `XXXXXX`(entity name) with your actual WandB project, experiment, and entitiy names.

```python
wandb_project_name = "ooooo"
wandb_exp_name = "???"
wandb.init(config = args,
           project = wandb_project_name,
           name = wandb_exp_name,
           entity = "xxxxx",
           settings=wandb.Settings(start_method='fork'))
```

## Usage

Choose schedulers for the learning rate, optimizer beta values, and batch size with the command-line options. For example:

```bash
python3 main.py \
  --lr_sche_name="constant" \
  --beta_1_sche_name="step_decay" \
  --beta_2_sche_name="constant" \
  --batch_sche_name="exp"
```
