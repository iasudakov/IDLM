set -euo pipefail
export PATH="/conda/envs/main/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python -m pip install \
  "transformers==4.55.4" \
  "ipdb==0.13.13" \
  "hydra-core==1.3.2" \
  "lightning==2.2.1" \
  "pytorch-lightning==2.2.1"

if printenv PYTHONPATH >/dev/null; then
  export PYTHONPATH="$SOURCE_CODE_PATH:$SOURCE_CODE_PATH/code:$PYTHONPATH"
else
  export PYTHONPATH="$SOURCE_CODE_PATH:$SOURCE_CODE_PATH/code"
fi

cd -- "$SOURCE_CODE_PATH/code"
python -V
command -v python
nvidia-smi

# Connect the source run's output state to this block's snapshot/state input.
# Make three launches, changing only the value below and the connected state:
# 5e0ee1da = MDLM -> CDLM; dbb3b25d = CDLM -> MMD; 8cb53b89 = MMD -> CDLM.
# Keep the same two offline asset inputs used in training; do not move weights.
# Request one GPU in the Nirvana block; use --devices 4 for a four-GPU block.
CDLM_EVAL_SOURCE_RUN=5e0ee1da \
bash "$SOURCE_CODE_PATH/code/scripts/eval_cdlm_checkpoints_nirvana.sh" \
  --devices 1 \
  "model_weights_dir=$INPUT_PATH/tinygsm_mmd_offline_bundle" \
  "eval.gsm8k_data_path=$INPUT_PATH/nirvana_checkpoints/data/gsm8k_test.json"
