"""Explicit entry point for the new archive implementation's paper ablations."""
from pathlib import Path

import hydra

from paper_objective import install

install()
from main import main as train_main


# Calling the imported main's Hydra wrapper resolves configs as pkg://configs.
# Use an absolute filesystem path so configs need not be a Python package.
@hydra.main(version_base=None,
            config_path=str(Path(__file__).resolve().parents[2] / 'configs'),
            config_name='config')
def main(config):
    return train_main.__wrapped__(config)


if __name__ == '__main__':
    main()
