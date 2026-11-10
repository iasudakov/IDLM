set -euo pipefail

# Use the same Python environment for installation and training.
export PATH="/conda/envs/main/bin:$PATH"

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

python -m pip install \
  "transformers==4.55.4" \
  "ipdb==0.13.13" \
  "hydra-core==1.3.2" \
  "lightning==2.2.1" \
  "pytorch-lightning==2.2.1"

# Reuse the inputs from the MMD job.
mv "$INPUT_PATH/nirvana_checkpoints/cache_dir" "$SOURCE_CODE_PATH/code/"
mv "$INPUT_PATH/nirvana_checkpoints/data" "$SOURCE_CODE_PATH/code/"
mv "$INPUT_PATH/nirvana_checkpoints/checkpoints" "$SOURCE_CODE_PATH/code/"
mv "$INPUT_PATH/tinygsm_mmd_offline_bundle" "$SOURCE_CODE_PATH/code/cache_dir/"

# Load the mounted Hugging Face token without printing it.
touch "$HF_HOME/token"
chmod 600 "$HF_HOME/token"
cat /slot/sandbox/d/secret/* > "$HF_HOME/token"

if printenv PYTHONPATH >/dev/null; then
  export PYTHONPATH="$SOURCE_CODE_PATH:$SOURCE_CODE_PATH/code:$PYTHONPATH"
else
  export PYTHONPATH="$SOURCE_CODE_PATH:$SOURCE_CODE_PATH/code"
fi

cd -- "$SOURCE_CODE_PATH/code"
mkdir -p outputs

python -V
command -v python
nvidia-smi
python -c 'import torch, flash_attn; print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "flash_attn:", flash_attn.__version__); print("GPU:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0)); print("compiled architectures:", torch.cuda.get_arch_list())'

# Connect the completed CDLM state to this block's snapshot input.
# Select the namespace used by the previous CDLM launch.
CDLM_SOURCE_NAMESPACE=cdlm_tinygsm_ancestral_sweep \
bash "$SOURCE_CODE_PATH/code/scripts/train_mmd_from_cdlm_tinygsm_nirvana.sh" \
  'logger.name=cdlm to mmd tinygsm 4xB300 paper recipe'
