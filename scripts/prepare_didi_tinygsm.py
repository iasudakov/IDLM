"""Explicit TinyGSM cache preparation; training never rebuilds shared data."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--data-path', help='Local JSON with question/code records')
    parser.add_argument('--allow-download', action='store_true',
                        help='Explicitly allow downloading TinyGSM/TinyGSM')
    parser.add_argument('--tokenizer-local-files-only', action='store_true')
    parser.add_argument('--length', type=int, default=512)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--separator', default=r'\n',
                        help='Default is literal backslash-n, matching reference checkpoints')
    parser.add_argument('--val-ratio', type=float, default=0.01)
    parser.add_argument('--val-seed', type=int, default=42)
    parser.add_argument('--train-on-pad', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.data_path and not args.allow_download:
        parser.error('provide --data-path or explicitly enable --allow-download')
    if args.length < 2 or args.workers < 0 or not 0 < args.val_ratio < 1:
        parser.error('need length>=2, workers>=0 and 0<val-ratio<1')
    import hydra
    from didi_repro.runtime import UPSTREAM_ROOT
    from didi_repro.tinygsm import register_configs
    from didi_repro.tinygsm_data import get_tokenizer, prepare_cache
    register_configs()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(UPSTREAM_ROOT / 'configs')):
        config = hydra.compose(config_name='config', overrides=['data=tiny-gsm'])
    config.data.cache_dir = str(Path(args.cache_dir).resolve())
    config.data.data_path = str(Path(args.data_path).resolve()) if args.data_path else None
    config.data.tokenizer_local_files_only = args.tokenizer_local_files_only
    config.data.separator = args.separator
    config.data.val_ratio = args.val_ratio
    config.data.val_seed = args.val_seed
    config.data.train_on_pad = args.train_on_pad
    config.model.length = args.length
    config.loader.num_workers = args.workers
    tokenizer = get_tokenizer(config)
    print(prepare_cache(config, tokenizer, allow_download=args.allow_download), flush=True)


if __name__ == '__main__':
    main()
