"""Gradient Moment metric (Hoogeboom et al., 2026, arXiv:2603.20155).

PyTorch implementation. All public symbols are accessible at the top level.

VENDORED from https://github.com/ehoogeboom/gradient-moment at commit
6ce7452cb38c2161ccbc9c5568afb38d4a0599be (v0.2.0), MIT licensed -- see the
LICENSE beside this file. Copied in rather than depended on because runs here
are packaged as `$SOURCE_CODE_PATH/code`, where a pip-installed package outside
the repo does not exist; `eval.owt_gm` would then skip silently on every cluster
run while working locally. It is 833 lines and needs only torch and numpy, both
already required.

Unmodified except for its packaging metadata, which is not copied: upstream
declares `requires-python = ">=3.10"` while this environment is pinned to 3.9 by
the rest of its stack. That gate was the only barrier -- every module here
already carries `from __future__ import annotations`, so the PEP 604 unions
never evaluate at runtime, all modules compile under 3.9, and none of the
3.10-only runtime features are used.
"""

from __future__ import annotations

from gradient_moment.core import (
    BatchPair,
    ForwardFn,
    GMEstimate,
    LossFn,
    diff_grad,
    gradient_moment_estimate,
    tree_inner_product,
)
from gradient_moment.metrics.base import Metric, evaluate, running_summary
from gradient_moment.metrics.generative_llm import make_generative_llm_metrics
from gradient_moment.metrics.sample_entropy import make_sample_entropy_metric
from gradient_moment.stats import (
    RunningStats,
    init_stats,
    stats_mean,
    stats_stderr,
    update_stats,
    update_stats_many,
)

__version__ = "0.2.0"

__all__ = [
    "BatchPair",
    "ForwardFn",
    "GMEstimate",
    "LossFn",
    "Metric",
    "RunningStats",
    "diff_grad",
    "evaluate",
    "gradient_moment_estimate",
    "init_stats",
    "make_generative_llm_metrics",
    "make_sample_entropy_metric",
    "running_summary",
    "stats_mean",
    "stats_stderr",
    "tree_inner_product",
    "update_stats",
    "update_stats_many",
]
