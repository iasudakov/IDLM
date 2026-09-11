"""Print True/False: does this checkpoint carry the distillation-only modules?

A checkpoint written by a distillation run stores the frozen `teacher` and the
`fake_model` critic alongside the student `backbone`, so the Lightning module
must be built with `adversarial_distill.is_distill=True` for the state_dict to
match. Released student checkpoints ship `backbone` alone. Only the key names
are needed here, so tensors are memory-mapped rather than read.
"""
import os
import pickle
import sys
import types

# The checkpoint pickle references this repo's own modules by name, and
# `sys.path[0]` is this script's directory rather than the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


class _Stub:
  """Inert stand-in for objects whose unpickling is library-version bound.

  Some checkpoints pickle a live `tokenizers.Tokenizer`, which refuses to
  deserialize against a different `tokenizers` build. Those objects are
  irrelevant to the key names, so they are replaced instead of reconstructed.
  """

  def __init__(self, *args, **kwargs):
    pass

  def __setstate__(self, state):
    pass


class _Unpickler(pickle.Unpickler):

  def find_class(self, module, name):
    if module.split('.')[0] in ('tokenizers', 'transformers'):
      return _Stub
    return super().find_class(module, name)


_shim = types.SimpleNamespace(
  __name__='_stub_pickle',
  Unpickler=_Unpickler,
  load=lambda f, **kw: _Unpickler(f, **kw).load(),
  loads=pickle.loads, dump=pickle.dump, dumps=pickle.dumps,
  HIGHEST_PROTOCOL=pickle.HIGHEST_PROTOCOL)


def main():
  ckpt = torch.load(sys.argv[1], map_location='cpu', mmap=True,
                    pickle_module=_shim, weights_only=False)
  print('True' if any(k.startswith(('teacher.', 'fake_model.'))
                      for k in ckpt['state_dict']) else 'False')


if __name__ == '__main__':
  main()
