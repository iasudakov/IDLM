set -euo pipefail
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

if printenv PYTHONPATH >/dev/null; then
  export PYTHONPATH="$SOURCE_CODE_PATH:$SOURCE_CODE_PATH/code:$PYTHONPATH"
else
  export PYTHONPATH="$SOURCE_CODE_PATH:$SOURCE_CODE_PATH/code"
fi

cd -- "$SOURCE_CODE_PATH/code"
python -V
command -v python
nvidia-smi

# No CDLM training snapshot is required. A prior MMD -> CDLM snapshot can reuse
# its published MMD asset. Keep the same tokenizer/data inputs used in training.
# The published model is public; only its missing weights may be downloaded.
# Request one GPU in the Nirvana block; use --devices 4 for a four-GPU block.
bash "$SOURCE_CODE_PATH/code/scripts/eval_mmd_checkpoint_nirvana.sh" \
  --devices 1 \
  "model_weights_dir=$INPUT_PATH/tinygsm_mmd_offline_bundle" \
  "eval.gsm8k_data_path=$INPUT_PATH/nirvana_checkpoints/data/gsm8k_test.json" \
  'logger.name=mdlm to mmd published checkpoint dual norm ancestral'
