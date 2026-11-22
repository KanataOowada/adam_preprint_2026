# On the Convergence of Vanilla Adam under Heavy-Tailed Noise and Index Sampling

## Abstruct
Understanding how stochastic gradient noise affects the convergence of Adaptive Moment Estimation (Adam) is an important problem in stochastic optimization, particularly when the noise exhibits heavy-tailed behavior. In this paper, we investigate the convergence of vanilla Adam in both the expected risk and the empirical risk minimization, focusing on the effects of heavy-tailed population noise and index-sampling noise. For the empirical risk minimization, we establish convergence guarantees under H\"older smoothness, showing that heavy-tailed population noise can affect the convergence bounds through dataset-dependent constants that may increase with the dataset size. For the expected risk minimization, we firstly establish both asymptotic and non-asymptotic convergence guarantees for vanilla Adam under H\"older smoothness and bounded $\mathfrak{p}$-th moment assumptions, allowing for infinite-variance stochastic gradient noise. In particular, we derive explicit convergence rates in terms of the expected squared gradient norm and show that the convergence rates deteriorate as $\mathfrak{p}$ decreases. Numerical experiments further support these theoretical findings and provide empirical evidence for the relevance of the sufficient conditions underlying our convergence analysis.

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
For experiment4heavy.py:

```bash
python3 -m pip install numpy matplotlib
python3 experiment4heavy.py          # Generate convergence plots
python3 experiment4heavy.py heatmap  # Generate the heatmap
python3 experiment4heavy.py all      # Run both experiments
```
