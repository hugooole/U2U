# REUSE Compliance

This project complies with the REUSE Specification 3.0 for clear licensing and copyright information.

## Copyright Headers

All source files include:
- **SPDX License Identifier**: `Apache-2.0`
- **Copyright**: Copyright (C) 2025 zhiguo

Example header in Python files:
```python
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
```

## License Location

The full Apache-2.0 license text is located in:
- `LICENSES/Apache-2.0.txt` - SPDX-compliant license text
- `LICENSE` - Root license file

## REUSE Configuration

- `.reuse/dep5` - REUSE dependency file with copyright and license declarations
- `LICENSES/` - Directory containing all license files

## Verification

To verify REUSE compliance:
```bash
reuse lint
```

## Pre-commit Hook

The `reuse` hook in `.pre-commit-config.yaml` automatically checks compliance on commit:
```bash
pre-commit run reuse --all-files
```
