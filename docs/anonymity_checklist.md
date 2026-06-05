# Anonymity Checklist

## Removed artifacts

The anonymous copy excludes:

- raw datasets and generated partitions under `data/<dataset>/`
- checkpoints and model weights (`*.pt`, `*.pth`)
- generated partitions (`*.pkl`)
- NumPy data dumps (`*.npy`, `*.npz` except small expected result CSVs are used instead)
- Hydra outputs, logs, tensorboard/runs/W&B directories
- cache folders such as `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ipynb_checkpoints`
- personal notebooks, local paper-writing folders, image packs, and temporary web/GPT packets

## Audit commands

Run these from the release root:

```bash
python scripts/audit_budget.py --root .

# Personal path and host patterns
git grep -n "/""home/" || true
git grep -n "C:""/" || true
git grep -n "C:""\\" || true
git grep -n "W""andB" || true

# Replace the placeholders below with the actual strings to verify before submission.
git grep -n "AUTHOR_NAME_TO_CHECK" || true
git grep -n "INSTITUTION_TO_CHECK" || true

# Email-like strings. This is preferred over raw at-sign checks because Python decorators use that character.
git grep -n -E "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}" || true
```

A raw at-sign grep is not used as a blocking check because Python decorators are expected code syntax, not identifying information.
