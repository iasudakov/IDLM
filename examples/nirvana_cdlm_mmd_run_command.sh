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

# Existing MMD inputs. The optional packaged MMD asset is stored separately
# under code/cdlm_model_assets, so these moves retain their existing layout.
mv "$INPUT_PATH/nirvana_checkpoints/cache_dir" "$SOURCE_CODE_PATH/code/"
mv "$INPUT_PATH/nirvana_checkpoints/data" "$SOURCE_CODE_PATH/code/"
mv "$INPUT_PATH/nirvana_checkpoints/checkpoints" "$SOURCE_CODE_PATH/code/"
mv "$INPUT_PATH/tinygsm_mmd_offline_bundle" "$SOURCE_CODE_PATH/code/cache_dir/"

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

# Resolve MMD once (bundled/input/cached/snapshot weights, then HF download),
# restore or collect trajectories, and train/evaluate on four GPUs.
CDLM_PRESET=cdlm_tinygsm_mmd \
bash "$SOURCE_CODE_PATH/code/scripts/train_cdlm_tinygsm_nirvana.sh" \
  'logger.name=mmd to cdlm tinygsm 4xB300 16 epochs'
