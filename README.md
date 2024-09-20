# AdversarialGenAIDetector
Project for DL4NLP at UVA (2024/2025)

## 1. Running on Lisa / Snellius

Install environment using
```bash
sbatch install_environment.job
```

Run interactive session
```bash
srun --partition=gpu --gpus=1 --ntasks=1 --cpus-per-task=18 --time=04:00:00 --pty bash -i
```

And later
```bash
module purge
module load 2022
module load Anaconda3/2022.05

source activate dl4nlp_gpu
```