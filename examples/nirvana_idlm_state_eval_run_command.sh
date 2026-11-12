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

# Source code: tinygsm-upstream-repro with the new evaluator.
# Connect run 1be7c4ce-1533-4a28-92a0-ba4de568d9f5 output state to the
# ordinary snapshot/state input. Keep the two offline asset inputs below.
# Request TWO GPUs. This state contains optimizer step 5000, NOT 5500.
bash "$SOURCE_CODE_PATH/code/scripts/eval_idlm_state_nirvana.sh" \
  --step 5000 \
  --devices 2 \
  --thresholds 0.99 0.995 \
  --model-weights-dir "$INPUT_PATH/tinygsm_mmd_offline_bundle" \
  --data-path "$INPUT_PATH/nirvana_checkpoints/data/gsm8k_test.json" \
  --name 'IDLM-REINFORCE step5000 FP32 confidence extension'

# Add --plan-only to inspect checkpoint selection without GPU evaluation.
# To remeasure the old endpoint as an overlap check:
#   --thresholds 0.98 0.99 0.995
# --step 5500 deliberately fails unless a different connected state contains
# an actual checkpoint whose saved global_step equals 5500.
