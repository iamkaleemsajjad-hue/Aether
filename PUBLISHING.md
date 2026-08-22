# How to Publish `aether-runtime` to PyPI

This guide explains how to make the package installable by anyone on earth via:
```bash
pip install aether-runtime
```

Currently the package is only installed as an **editable local install** on your machine. This guide walks you through publishing it to the real Python Package Index (PyPI) so anyone can install it.

---

## Overview

```
Your Machine → GitHub → GitHub Actions → PyPI → pip install aether-runtime ✅
```

The process uses **OIDC Trusted Publisher** — no API keys or secrets needed. GitHub Actions authenticates directly with PyPI using a short-lived token.

---

## Part 1 — One-Time Setup (Do This Once)

### Step 1: Create a PyPI Account

1. Go to **https://pypi.org/account/register/** and create an account
2. Verify your email address
3. Enable **Two-Factor Authentication (2FA)** — required for publishing

### Step 2: Create a TestPyPI Account

TestPyPI is a safe sandbox to test publishing before the real PyPI.

1. Go to **https://test.pypi.org/account/register/**
2. Create a separate account (or same email)
3. Verify your email

---

### Step 3: Configure Trusted Publisher on TestPyPI

> Trusted Publisher = GitHub Actions publishes directly without an API key. It's safer and simpler.

1. Log in to **https://test.pypi.org**
2. Go to your account → **"Your projects"** → **"Add a new project"**  
   *(Or if the project name `aether-runtime` is already taken, use a different name like `aether-runtime-dev`)*
3. Click **"Publishing"** in the left sidebar
4. Click **"Add a new pending publisher"**
5. Fill in the form:

   | Field | Value |
   |-------|-------|
   | **PyPI Project Name** | `aether-runtime` |
   | **Owner** | `iamkaleemsajjad-hue` |
   | **Repository name** | `Aether` *(your GitHub repo name)* |
   | **Workflow filename** | `release.yml` |
   | **Environment name** | `testpypi` |

6. Click **"Add"**

---

### Step 4: Configure Trusted Publisher on Real PyPI

1. Log in to **https://pypi.org**
2. Go to **https://pypi.org/manage/account/publishing/**
3. Click **"Add a new pending publisher"**
4. Fill in the form:

   | Field | Value |
   |-------|-------|
   | **PyPI Project Name** | `aether-runtime` |
   | **Owner** | `iamkaleemsajjad-hue` |
   | **Repository name** | `Aether` |
   | **Workflow filename** | `release.yml` |
   | **Environment name** | `pypi` |

5. Click **"Add"**

---

### Step 5: Create GitHub Environments

GitHub environments are used by the workflow for deploy protection rules.

1. Go to your GitHub repo: **https://github.com/iamkaleemsajjad-hue/Aether**
2. Click **Settings** → **Environments** → **New environment**
3. Create environment named: **`testpypi`** (no protection rules needed)
4. Create another environment named: **`pypi`**
   - Optionally add **"Required reviewers"** (your own username) for extra safety
5. That's it — no secrets needed because we use Trusted Publisher

---

### Step 6: Enable GitHub Pages

This fixes the failing `github-pages` deployment:

1. Go to **https://github.com/iamkaleemsajjad-hue/Aether/settings/pages**
2. Under **"Source"** → select **"GitHub Actions"**
3. Click **Save**

Now the docs workflow can deploy to `https://iamkaleemsajjad-hue.github.io/Aether/`

---

## Part 2 — Publish a Release (Do This Each Time)

### Step 1: Verify the build locally first

```powershell
# In your project directory
cd "C:\Users\pc\Desktop\Aether Runtime"

# Install build tools
pip install build twine

# Build the package
python -m build

# Check the output — you should see two files:
dir dist\

# Verify the package metadata
python -m twine check --strict dist\*
```

Expected output from `twine check`:
```
Checking dist/aether_runtime-1.2.0-py3-none-any.whl: PASSED
Checking dist/aether_runtime-1.2.0.tar.gz: PASSED
```

If you see `PASSED` — you're ready to publish.

---

### Step 2: Update the version number

Open [`pyproject.toml`](pyproject.toml) and bump the version:

```toml
[project]
name = "aether-runtime"
version = "1.2.0"   # current release
```

> Follow [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`
> - `PATCH` (1.2.0 → 1.2.1): Bug fixes
> - `MINOR` (1.2.0 → 1.3.0): New features, backwards compatible
> - `MAJOR` (1.2.0 → 2.0.0): Breaking changes

---

### Step 3: Commit the version bump

```powershell
git add pyproject.toml
git commit -m "release: prepare v1.2.0"
git push origin main
```

---

### Step 4: Create and push a version tag

> This is what triggers the GitHub Actions publish workflow.

```powershell
# Create a tag matching the version in pyproject.toml
git tag v1.2.0

# Push the tag to GitHub — this starts the publish workflow
git push origin v1.2.0
```

---

### Step 5: Watch the publish workflow run

1. Go to **https://github.com/iamkaleemsajjad-hue/Aether/actions**
2. You'll see **"Publish to PyPI"** workflow running
3. It will:
   - ✅ Build the wheel and sdist
   - ✅ Publish to TestPyPI
   - ✅ Publish to real PyPI
   - ✅ Create a GitHub Release with the wheel attached

---

### Step 6: Verify it worked

Wait ~2 minutes after the workflow completes, then:

```bash
# Test install in a fresh environment (use a temp venv)
python -m venv /tmp/test-aether
/tmp/test-aether/Scripts/activate   # Windows: /tmp/test-aether\Scripts\activate

pip install aether-runtime
aether version
```

Your package is now public on **https://pypi.org/project/aether-runtime/**

---

## Part 3 — Troubleshooting

### "Name already taken on PyPI"

The name `aether-runtime` might already be registered by someone else on PyPI.

```bash
# Check if the name is taken
pip index versions aether-runtime 2>&1 || echo "Name is free"
```

If it's taken, change the name in `pyproject.toml`:
```toml
name = "aether-runtime-engine"   # or another unique name
```

Then re-configure the Trusted Publisher with the new name.

---

### "Trusted publisher not configured" error in Actions

This means you haven't completed Step 3/4 above. The workflow says:
```
Error: 403 Forbidden: Invalid or non-existent authentication information.
```

Go back to PyPI → Publishing → verify the Trusted Publisher settings match exactly:
- Owner: `iamkaleemsajjad-hue`
- Repo: `Aether` (case-sensitive)
- Workflow: `release.yml`
- Environment: `pypi` or `testpypi`

---

### "Version already exists" error

You can't re-upload the same version number. Bump the version in `pyproject.toml` and create a new tag.

---

### Workflow not triggering

Make sure you push the **tag**, not just the commit:
```bash
git push origin v1.2.0    # ← pushes the tag — triggers workflow
git push origin main      # ← pushes the commit — does NOT trigger workflow
```

---

## Part 4 — After Publishing

### What anyone can now do

```bash
# Install from PyPI
pip install aether-runtime

# Install with GPU support
pip install "aether-runtime[transformers-frontend]"

# Install in a Kaggle notebook
!pip install aether-runtime

# Install in Google Colab
!pip install aether-runtime
```

### Your package page

After publishing, your package will be at:
```
https://pypi.org/project/aether-runtime/
```

---

## Summary Checklist

- [ ] Create PyPI account + enable 2FA: https://pypi.org/account/register/
- [ ] Create TestPyPI account: https://test.pypi.org/account/register/
- [ ] Add Trusted Publisher on **TestPyPI** (owner=`iamkaleemsajjad-hue`, repo=`Aether`, workflow=`release.yml`, env=`testpypi`)
- [ ] Add Trusted Publisher on **PyPI** (same settings, env=`pypi`)
- [ ] Create GitHub environments: `testpypi` and `pypi` in repo Settings → Environments
- [ ] Enable GitHub Pages: repo Settings → Pages → Source = "GitHub Actions"
- [ ] Bump version in `pyproject.toml`
- [ ] Commit + push
- [ ] Create + push tag: `git tag v1.2.0 && git push origin v1.2.0`
- [ ] Watch workflow at: https://github.com/iamkaleemsajjad-hue/Aether/actions
- [ ] Verify: `pip install aether-runtime` works from any machine
