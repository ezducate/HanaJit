# Publishing HanaJit to PyPI

This is the complete, step-by-step process to publish HanaJit to the Python
Package Index (PyPI). It covers both the **recommended** path (automated,
via GitHub Actions + Trusted Publishing — no API tokens to manage) and the
**manual** path (build and upload from your machine). Read the whole document
once before starting.

HanaJit is a **pure-Python** package (its only dependency, `llvmlite`, ships
its own binary wheels), so publishing is simple: one universal wheel plus a
source distribution, no per-platform build matrix required.

---

## 0. Prerequisites

- The repository is pushed to `https://github.com/ezducate/HanaJit`.
- The version in `pyproject.toml` and `hanajit/__init__.py` is the one you
  intend to release (they must match — see step 2).
- The test suite passes: `python -m pytest tests/ -q`.
- You have (or will create) accounts on **PyPI** and **TestPyPI**:
  - PyPI: https://pypi.org/account/register/
  - TestPyPI: https://test.pypi.org/account/register/
  - Enable two-factor authentication on both (PyPI requires it for uploads).

> **Name availability:** the distribution name `hanajit` was free at the time
> of writing. PyPI names are first-come; confirm at
> https://pypi.org/project/hanajit/ (a 404 means it's still available). If it
> has been taken, change `name = "..."` in `pyproject.toml` to an available
> name (e.g. `hanajit-llvm`) before publishing.

---

## 1. One-time: verify packaging metadata

Everything PyPI displays comes from `pyproject.toml`. Confirm it contains:

- `name`, `version`, `description`, `readme = "README.md"`
- `requires-python = ">=3.10"`
- `license` and the matching license classifier
- `authors`, `keywords`, `classifiers`
- `[project.urls]` pointing at the GitHub repo
- `dependencies = ["llvmlite>=0.42"]`

The `README.md` is rendered as the project's front page on PyPI, so make sure
it looks right in Markdown.

---

## 2. Bump and synchronize the version

PyPI **rejects re-uploads of an existing version** — every release needs a new
version number. Update it in **both** places and keep them identical:

- `pyproject.toml` → `version = "X.Y.Z"`
- `hanajit/__init__.py` → `__version__ = "X.Y.Z"`

Verify they agree:

```bash
python -c "import hanajit; print(hanajit.__version__)"
grep '^version' pyproject.toml
```

Follow semantic versioning: `0.20.0` → `0.20.1` (patch) / `0.21.0` (features)
/ `1.0.0` (stable API). While in alpha, keep the `Development Status :: 3 -
Alpha` classifier; move to `4 - Beta` / `5 - Production/Stable` when ready.

Update `CHANGELOG.md` with the release notes, commit, and tag:

```bash
git add pyproject.toml hanajit/__init__.py CHANGELOG.md
git commit -m "Release vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

---

## 3. Build the distributions

Install the build tool and produce the wheel + sdist:

```bash
python -m pip install --upgrade build
python -m build
```

This creates `dist/`:

```
dist/
  hanajit-X.Y.Z-py3-none-any.whl     # the wheel (what pip installs)
  hanajit-X.Y.Z.tar.gz               # the source distribution
```

`py3-none-any` confirms it's a pure-Python, platform-independent wheel — good.

**Validate before uploading:**

```bash
python -m pip install --upgrade twine
python -m twine check dist/*          # checks metadata + README rendering
```

Both files should report `PASSED`. Optionally inspect the wheel contents:

```bash
python -m zipfile -l dist/hanajit-X.Y.Z-py3-none-any.whl
```

Make sure it contains the `hanajit/` package and **not** stray files
(`__pycache__`, tests you didn't intend to ship, `.pytest_cache`). The
`.gitignore` and setuptools defaults handle most of this; if unwanted files
appear, add a `MANIFEST.in` or tighten `[tool.setuptools.packages.find]`.

---

## 4. Test on TestPyPI first (strongly recommended)

TestPyPI is a throwaway copy of PyPI. Upload there and do a real install to
catch problems before the real release.

```bash
python -m twine upload --repository testpypi dist/*
```

It will prompt for credentials (see step 5 for token setup). Then install
from TestPyPI in a clean virtual environment — note the extra index, because
TestPyPI does not host `llvmlite`, so dependencies come from real PyPI:

```bash
python -m venv /tmp/testenv
/tmp/testenv/bin/pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  hanajit==X.Y.Z

/tmp/testenv/bin/python -c "import hanajit; print(hanajit.__version__)"
```

If that imports and runs, you're ready for the real thing.

---

## 5. Publish to PyPI

You have two paths. **Path A (Trusted Publishing) is recommended** — no
long-lived tokens, nothing secret to store.

### Path A — Automated via GitHub Actions + Trusted Publishing (recommended)

PyPI can trust releases that come from a specific GitHub repository and
workflow, using short-lived OpenID Connect tokens. No API token is ever
created or stored.

**A1. Register the "pending publisher" on PyPI** (do this once, before the
first release):

1. Log in to PyPI → your account → **Publishing** →
   **Add a new pending publisher**.
2. Fill in:
   - **PyPI project name:** `hanajit`
   - **Owner:** `ezducate`
   - **Repository name:** `HanaJit`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi` (optional but recommended; see A3)
3. Save. (Do the same on TestPyPI under its Publishing settings if you want
   automated TestPyPI uploads.)

**A2. The workflow is already in the repo** at
`.github/workflows/publish.yml`. It builds the distributions and uploads them
using the `pypa/gh-action-pypi-publish` action with OIDC. Confirm it targets
the right trigger — typically publishing when you push a version tag or create
a GitHub Release. A minimal, correct version looks like:

```yaml
name: Publish to PyPI
on:
  release:
    types: [published]        # fires when you publish a GitHub Release
permissions:
  contents: read
jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    environment: pypi          # matches the pending-publisher environment
    permissions:
      id-token: write          # REQUIRED for Trusted Publishing (OIDC)
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python -m pip install --upgrade build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
        # no username/password/token needed — OIDC handles auth
```

**A3. (Recommended) Add a protected environment.** In the GitHub repo →
Settings → Environments → create `pypi`. You can require manual approval
before the publish job runs, so a release can't go out by accident.

**A4. Release.** Create a GitHub Release (Releases → Draft a new release →
choose the `vX.Y.Z` tag → publish). The workflow runs, builds, and uploads to
PyPI automatically. Watch it in the **Actions** tab.

### Path B — Manual upload from your machine

If you prefer to upload by hand (or Trusted Publishing isn't set up yet):

**B1. Create an API token** on PyPI → account → **API tokens** → Add token.
Scope it to the `hanajit` project once the project exists (for the very first
upload you may need an account-wide token, then replace it with a
project-scoped one afterward).

**B2. Store it** in `~/.pypirc` (keep this file private, `chmod 600`):

```ini
[pypi]
  username = __token__
  password = pypi-AgEId...your-token...

[testpypi]
  username = __token__
  password = pypi-AgEId...your-testpypi-token...
```

Alternatively pass it inline: `twine upload -u __token__ -p pypi-... dist/*`.

**B3. Upload:**

```bash
python -m twine upload dist/*
```

---

## 6. Verify the release

```bash
# fresh environment, install from real PyPI
python -m venv /tmp/verifyenv
/tmp/verifyenv/bin/pip install hanajit==X.Y.Z
/tmp/verifyenv/bin/python -c "import hanajit; print(hanajit.__version__)"
```

Then check the project page at `https://pypi.org/project/hanajit/`:
the README should render, the metadata and links should be correct, and the
version should be listed.

---

## 7. After releasing

- **You cannot overwrite or re-upload a version.** If you find a bug
  immediately, publish a new patch version (e.g. `X.Y.Z+1`). You can *yank* a
  broken release on PyPI (it stays installable only by exact pin, and is
  hidden from new installs) but you cannot replace its files.
- Update the README's install section once `pip install hanajit` works
  (remove the "not yet published" note).
- Announce / link the release from the GitHub Release notes and the landing
  page in `site/`.

---

## Quick reference (the whole flow)

```bash
# 1. bump version in pyproject.toml AND hanajit/__init__.py (must match)
# 2. update CHANGELOG.md, commit, tag
git commit -am "Release vX.Y.Z" && git tag vX.Y.Z && git push origin main --tags

# 3. build + validate
python -m build
python -m twine check dist/*

# 4. (recommended) test on TestPyPI
python -m twine upload --repository testpypi dist/*

# 5a. publish via GitHub Release (Trusted Publishing) — just publish the Release
#     on GitHub and let Actions do it, OR
# 5b. publish manually
python -m twine upload dist/*

# 6. verify
pip install hanajit==X.Y.Z
```

---

## Troubleshooting

- **`File already exists` on upload** — the version was already published.
  Bump the version; you can't re-upload.
- **README doesn't render on PyPI** — run `twine check dist/*`; usually a
  Markdown issue or a missing `readme = "README.md"` in `pyproject.toml`.
- **`invalid or non-existent authentication`** (manual path) — the username
  must be the literal string `__token__` and the password the full token
  including the `pypi-` prefix.
- **Trusted Publishing fails with an OIDC error** — confirm the workflow has
  `permissions: id-token: write`, the job's `environment:` matches the
  pending-publisher environment name, and the repo owner/name/workflow
  filename exactly match what you registered on PyPI.
- **Dependency `llvmlite` not found when installing from TestPyPI** — add
  `--extra-index-url https://pypi.org/simple/` so pip can fetch it from real
  PyPI (TestPyPI doesn't mirror it).
- **Unwanted files in the wheel** — inspect with `python -m zipfile -l
  dist/*.whl`; add a `MANIFEST.in` or adjust the setuptools package
  discovery to exclude them.
