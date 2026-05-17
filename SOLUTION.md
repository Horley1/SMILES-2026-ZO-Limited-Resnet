# SMILES-2026. Kalmykov Maksim

## Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run evaluation

```bash
python validate.py \
    --data_dir ./data \
    --batch_size 32 \
    --n_batches 256 \
    --output results.json
```

256 steps * 32 batch size = 8192 samples (exactly the maximum allowed budget)


## Solution

The main problem with zero-order optimization is the high dimensionality of the classification head.
 
To solve this, I implemented **Fixed-B LoRA + SPSA + Adam**.

### Optimizer: 
*   **Fixed-B LoRA & SPSA:**
    Instead of optimizing the full weight matrix, I used a LoRA. 
    Crucially, I initialized $B \sim \mathcal{N}(0, \sigma^2)$ and **froze it**. I initialized $A = 0$ and applied SPSA perturbations only to $A$. 
    If we perturb both $A$ and $B$ simultaneously (as in standard LoRA), the change in effective weights is $O(\epsilon^2)$, which destroys the gradient signal. By fixing $B$, the perturbation remains $O(\epsilon)$.
*   **Adam update:**
    SPSA pseudo-gradients are inherently noisy. I implemented manual Adam moment tracking to smooth the updates. Vanilla SGD diverges.
### Head Init
*    Used `nn.init.xavier_uniform_` but explicitly scaled weights by `0.01`. A standard initialization yields large logits, meaning the optimizer wastes its tiny 256-step budget just moving the softmax outputs to a uniform distribution. Scaling by 0.01 provides a near-uniform probability distribution from step zero.
### Augmentations
*   Kept augmentations minimal (Resize, RandomHorizontalFlip). With only 256 optimization steps, heavy augmentations add unnecessary noise to the loss evaluations, destabilizing SPSA.

**What contributed most:**
The math fix in LoRA (fixing $B$ and perturbing only $A$) made optimization possible. The initialization scaling (`* 0.01`) provided the biggest immediate metric boost by drastically lowering the starting loss.

## Experiments

I ran extensive sweeps to tune the pseudo-gradient estimation and optimization parameters. Here is what failed or performed worse during the search:

*   **Evolutionary Strategies (ES) with Fitness Shaping:** 
    To make SPSA robust to loss outliers, I implemented rank-based fitness shaping. I stored all perturbation directions, ranked their loss differences ($f_{plus} - f_{minus}$), and assigned weights from -0.5 to +0.5 based on the rank. It gave practically no effect compared to simple averaging.
*   **High LoRA Ranks:** 
    Swept ranks from 4 to 128. Rank 4 is too restrictive (underfitting). Increasing the rank to 128 degraded performance. The subspace became too large, making the SPSA gradient variance fatal again.
*   **High Learning Rate:** 
    Adam needs aggressive steps here, but pushing `lr` further (e.g. beyond 0.1) caused severe overshooting and degraded the metric. 
*   **Uniform Perturbation for SPSA:** 
    Tested Uniform noise instead of Gaussian. Accuracy collapsed. Gaussian noise is strictly better for exploring this parameter space.
*   **Standard LoRA + SPSA:** 
    I initially tried optimizing both $A$ and $B$. The linear $O(\epsilon)$ terms cancel out, leaving only $O(\epsilon^2)$ noise. The loss didn't improve at all until I fixed $B$.

### Other runs

Here are the results from different configurations tested.

- **baseline_reproduce**: Top-1 Accuracy: 2.27% | Config: `lr=0.05, rank=16, spsa=100`
- **nspsa_200**: Top-1 Accuracy: 2.85% | Config: `lr=0.05, rank=16, spsa=200`
- **nspsa_50**: Top-1 Accuracy: 2.49% | Config: `lr=0.05, rank=16, spsa=50`
- **rank_32**: Top-1 Accuracy: 2.43% | Config: `lr=0.05, rank=32, spsa=100`
- **rank_4**: Top-1 Accuracy: 1.60% | Config: `lr=0.05, rank=4, spsa=100`
- **lr_0.1**: Top-1 Accuracy: 2.88% | Config: `lr=0.1, rank=16, spsa=100`
- **lr_0.01**: Top-1 Accuracy: 1.42% | Config: `lr=0.01, rank=16, spsa=100`
- **eps_1.0**: Top-1 Accuracy: 2.14% | Config: `lr=0.05, rank=16, spsa=100`
- **eps_0.1**: Top-1 Accuracy: 2.28% | Config: `lr=0.05, rank=16, spsa=100`
- **init_orthogonal**: Top-1 Accuracy: 3.33% | Config: `lr=0.05, rank=16, spsa=100`
- **init_xavier_small**: Top-1 Accuracy: 3.24% | Config: `lr=0.05, rank=16, spsa=100`
- **stratified**: Top-1 Accuracy: 2.40% | Config: `lr=0.05, rank=16, spsa=100`
- **no_fitness_shaping**: Top-1 Accuracy: 2.27% | Config: `lr=0.05, rank=16, spsa=100`
- **uniform_perturbation**: Top-1 Accuracy: 1.59% | Config: `lr=0.05, rank=16, spsa=100`
```