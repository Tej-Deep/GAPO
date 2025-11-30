# GAPO

This Repository contains the code for Group Advantage Preference Optimisation (GAPO).

## Setup
You can setup a conda env to run the scripts using the follwing commands:
```bash
conda create -n GAPO python=3.10.16
conda activate GAPO

pip install -r requirements.txt
pip intall flash-attn --no-build-isolation 
```

### Data
You can download the prm800k data for training the models from `https://github.com/openai/prm800k`. The processed benchmarks for evaluation are provided under `src/eval/eval_data`.

### Scripts
The bash scripts and notebooks required for data preparation, training and evaluation can be found under the respective folders in `./src`. 