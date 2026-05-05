# Fine-Tuning Neural Data Transformer with Multi-Task Masking on Spike-Sorted Neuropixels Data

## Overview 

This repository contains the local fine-tuning and evaluation pipeline used to adapt the IBL-MtM/NDT1 codebase to a single local control session. Large generated files such as raw data, preprocessed arrays, trained checkpoints, logs, and full result directories are not stored in GitHub.

## Setup Instructions

### Step 1: Clone Repository 
~~~bash
git clone <repo-url>
cd comp3931-finetune-mtm
~~~



### Step 2: Create The Python Environment

The original environment specification is provided in env.yaml

~~~ bash
conda env create -f env.yaml
conda activate <env-name>
~~~

On the HPC cluster used for this project, jobs were run with the site PyTorch module:

~~~ bash
module purge
module load pytorch/2.5.1
~~~

### Step 3: Update Paths 

All the code and scripts currently use placeholders. Before running preprocessing, training, or analysis, manually edit the paths in the relevant scripts/templates to match your filesystem. These include:

- Repository root
- Raw Kilosort directory
- Preprocessed data directory
- Results directory
- Logs directory
- Pretrained IBL-MtM checkpoint path

For preprocessing, edit:
~~~ bash
preprocess_control_session.py
~~~

Specifically update:
~~~ bash
KS_DIR = Path("/path/to/raw/control_session_kilosort4")
OUT_DIR = Path("/path/to/data/control_session_preprocessed_new")
~~~

For Slurm jobs, edit the relevant file under:
~~~ bash
scripts/templates/
~~~

The pretrained IBL-MtM checkpoint should point to:
~~~ bash 
checkpoints/models/ndt1_mtm_10/model_best.pt
~~~

The finetuned checkpoints from this project are not included due to file size. 
Results and evaluation metrics are saved in artifacts/ for each run instead 
(see Reproducibility Notes above).

The pretrained IBL-MtM checkpoint can be downloaded from HuggingFace. This model currently loads the 10-session one. 

### Step 4: Preprocess The Local Session

The preprocessing step converts raw Kilosort output into dense spike-count arrays used by the training scripts.

~~~ bash
python preprocess_control_session.py
~~~

Expected output files include:
~~~ bash
spikes_data.npy
time_attn_mask.npy
space_attn_mask.npy
spikes_timestamps.npy
spikes_spacestamps.npy
eid.npy
unit_ids.npy
neuron_regions.npy
split_indices.npz
metadata.json
~~~

### Step 5: Submit Training Jobs
Training jobs are launched through Slurm scripts or Slurm templates. Before submission, check that all paths, checkpoint locations, hyperparameters, and output directories are correct.

Example: 
~~~ bash
sbatch scripts/templates/train_mtm_combined.slurm.template
~~~

### Step 6: Training Runs

The main experiment families include:
| Experiment | Python script |
|---|---|
| IBL-MtM neuron-only fine-tuning | `src/train_single_session_local.py` |
| IBL-MtM causal-only fine-tuning | `src/train_single_session_local_causal.py` |
| IBL-MtM combined neuron+causal fine-tuning | `src/train_single_session_local_mtm_combined.py` |
| Direct NDT1 scratch | `src/train_single_session_local_ndt1_scratch.py` |
| Direct NDT1 scratch combined | `src/train_single_session_local_ndt1_scratch_combined.py` |
| Stitched NDT1 scratch | `src/train_single_session_local_ndt1_stitched_scratch.py` |
| Stitched NDT1 scratch combined | `src/train_single_session_local_ndt1_stitched_scratch_combined.py` |

For single-objective scratch runs, set the masking mode in the Slurm script:

~~~ bash
--masking-mode neuron
--mask-ratio 0.3
~~~

or 
~~~ bash
--masking-mode causal
--mask-ratio 0.6
~~~

For combined runs, use:
~~~ bash
--neuron-mask-ratio 0.3
--causal-mask-ratio 0.6
--causal-prob 0.5
~~~

### Step 7: Evaluation 
After training, evaluate a saved run by setting the run directory in the relevant evaluation Slurm script:
~~~ bash
RUN_DIR=results_20ms/<run_name>
~~~

Then submit:
~~~ bash
sbatch scripts/evaluate_loo.slurm
~~~

### Step 8: Analysis Scripts

The main analysis scripts are:
~~~ bash 
src/analyse_01_thesis_plots.py
src/analyse_03_generalisation.py
src/analyse_04_spike_groups.py
src/analyse_05_latent.py
src/analyse_05_latent_generalization.py
src/analyse_07_loo_comparison.py
~~~
These scripts read saved artifacts from results_20ms/ and generate summary tables and dissertation figures.

## Step 9: Files not added to GitHub
The following are intentionally excluded from version control:

~~~bash
logs/
results_20ms/
checkpoints/
data/**/*.npy
data/**/*.npz
*.pt
~~~

### Step 10: Reproducibility Notes

Each training run saves:

~~~ bash 
artifacts/run_metadata.json
artifacts/pretrained_load_report.json
artifacts/history.csv
artifacts/eval_metrics.json
~~~

Combined runs additionally save:

~~~ bash 
artifacts/combined_eval_summary.csv
artifacts/combined_eval_summary.json
artifacts/combined_generalization_gap.json
~~~

These files record the command-line arguments, model configuration, checkpoint-loading report, training history, and deterministic evaluation metrics.


Note: Hyperparameters tables for these runs can be referred to in Appendix C of the project file. 