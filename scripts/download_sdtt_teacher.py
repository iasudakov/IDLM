"""Download the declared SDTT Large teacher and record immutable file hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmd_sdtt_features import DEFAULT_MODEL, DEFAULT_REVISION


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--model', default=DEFAULT_MODEL)
  parser.add_argument('--revision', default=DEFAULT_REVISION)
  parser.add_argument('--local-files-only', action='store_true')
  args = parser.parse_args()
  from huggingface_hub import snapshot_download
  root = Path(snapshot_download(repo_id=args.model, revision=args.revision,
                               allow_patterns=['config.json', 'model.safetensors'],
                               local_files_only=args.local_files_only))
  out = args.output_dir.expanduser()
  out.mkdir(parents=True, exist_ok=True)
  import shutil
  files = []
  for name in ('config.json', 'model.safetensors'):
    source = root / name
    if not source.is_file():
      raise FileNotFoundError(f'SDTT teacher revision does not contain {name}')
    dest = out / name
    if source.resolve() != dest.resolve():
      shutil.copy2(source, dest)
    digest = hashlib.sha256()
    with dest.open('rb') as stream:
      for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        digest.update(chunk)
    files.append({'name': name, 'bytes': dest.stat().st_size, 'sha256': digest.hexdigest()})
  manifest = {'repo_id': args.model, 'requested_revision': args.revision,
              'resolved_commit': root.name, 'files': files}
  (out / 'sdtt_provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
  print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
  main()
