# On the Convergence of Vanilla Adam under Heavy-Tailed Noise and Index Sampling

## Abstruct
Understanding how stochastic gradient noise affects the convergence of Adam is an important problem in stochastic optimization, particularly when the noise exhibits heavy-tailed behavior. In this paper, we investigate the convergence of vanilla Adam in both the expected risk and the empirical risk minimization, focusing on the effects of heavy-tailed noise and its relationship to index sampling. For the expected risk minimization, we establish convergence guarantees under H\"older smoothness and bounded $\mathfrak{p}$-th moment assumptions, allowing stochastic gradient noise with infinite variance. Our analysis provides explicit upper bounds on the expected squared gradient norm and shows that the iteration-dependent convergence term decays more slowly as $\mathfrak{p}$ decreases. For the empirical risk minimization, we establish asymptotic and non-asymptotic convergence guarantees and demonstrate that heavy-tailed population noise can deteriorate convergence bounds through dataset-dependent constants as the dataset size increases. Furthermore, under Lipschitz smoothness, we formulate multistage Adam with practical learning rate schedules and demonstrate that increasing batch sizes can reduce index-sampling noise and improve convergence guarantees and SFO complexity. Numerical experiments support our theoretical findings on the effects of heavy-tailed noise and the computational benefits of increasing batch sizes.

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
