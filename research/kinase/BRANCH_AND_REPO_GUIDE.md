# Hyaline: Branch Strategy & Repo Organization

**Purpose:** Keep original Hyaline (GPCR) pristine while maintaining kinase work. Follow standard OSS practice (similar to Boltz).

---

## TL;DR — What To Do

1. **master** = original Hyaline (GPCR). Do not change.
2. **kinase-v1** = kinase work. Use a separate branch (not a folder).
3. **Fix before commit:** Rename `hyaline/data/` → `hyaline/loaders/` and fix imports, so `hyaline/data.py` (GPCR) still works.
4. **Push:** `git push -u origin kinase-v1` — then the branch appears on GitHub.

No second repo needed. No second folder for "Hyaline 1" — the original stays on `master`.

---

## Current State

| Item | Status |
|------|--------|
| **master** | Clean, original Hyaline (GPCR only). Commits: `bc6afdd` "update README" |
| **kinase-v1** | Same commit as master, but has **staged** changes (not yet committed) |
| **kinase-v1 on GitHub** | Not pushed — that's why you don't see it |
| **Original Hyaline** | Intact on master: `model_v2.py`, `data.py`, `predict.py`, GPCR only |

### ⚠️ Conflicts on kinase-v1

1. **Package shadowing:** The staged changes add `hyaline/data/` (a Python **package**). The original has `hyaline/data.py` (a **module**). In Python, `hyaline/data/` shadows `hyaline/data.py`, so `from hyaline.data import load_dataset_with_motifs` (used by `scripts/train.py`) would break — the data package doesn't export it.

2. **Broken import:** `hyaline/__init__.py` does `from .data import pdb_mining`, but `pdb_mining` doesn't exist in the data package — this will raise ImportError.

---

## Two Approaches

### Approach A: Branch-Only (Quick, Standard)

**Idea:** Keep `master` = original Hyaline. Use `kinase-v1` as a separate long-lived branch. Don’t merge to master until kinase is validated.

**What to do:**
1. Leave `master` as-is.
2. Commit and push `kinase-v1` with current changes.
3. On GitHub you’ll have:
   - `master` — GPCR Hyaline (default)
   - `kinase-v1` — GPCR + kinase + TF (separate branch)

**Pros:** Simple, no refactor  
**Cons:** kinase-v1 replaces the data layer; GPCR training may break on that branch

---

### Approach B: Additive Structure (Clean, Boltz-Style)

**Idea:** Keep original code paths working and add kinase as a subpackage. Single main branch later can have both.

**Structure:**
```
hyaline/
├── __init__.py          # Exports BOTH gpcr and kinase
├── model_v2.py          # Original GPCR (unchanged)
├── data.py             # Original (unchanged)
├── predict.py          # Original (unchanged)
├── ...
├── gpcr/               # (Optional) Group original GPCR if desired
├── kinase/             # Kinase-specific code
│   ├── __init__.py
│   ├── klifs_loader.py
│   └── ...
└── loaders/             # RENAMED from data/ to avoid shadowing data.py
    ├── tf_activation_data.py
    ├── pdb_loader.py
    └── ...
```

**Main fix:** Rename `hyaline/data/` → `hyaline/loaders/` (or `hyaline/pipelines/`) so it no longer shadows `hyaline/data.py`.

**Pros:** Original GPCR stays usable; kinase is additive  
**Cons:** Requires refactor and import updates

---

## Recommended Path (Pragmatic)

### Phase 1: Ship kinase-v1 (This Week)

1. **Rename the conflicting package** (avoids breaking GPCR on kinase branch):
   ```bash
   git checkout kinase-v1
   mv hyaline/data hyaline/loaders
   # Update all imports: hyaline.data.X → hyaline.loaders.X
   ```

2. **Update `.gitignore`** so large or local data stays out of the repo:
   ```
   klifs_cache/
   checkpoints/
   *.pt
   ```
   Keep `research/` in the repo (documentation).

3. **Commit and push:**
   ```bash
   git add .
   git commit -m "feat(kinase): Add kinase conformational selectivity pipeline

   - KLIFS loader, hybrid model, virtual screening scripts
   - TF activation and CryptoSite modules
   - Research docs, task assignments, product brief
   - Rename hyaline/data -> hyaline/loaders to preserve GPCR data.py"
   
   git push -u origin kinase-v1
   ```

4. **On GitHub:** Both branches visible. Default remains `master` (GPCR).

### Phase 2: Optional Restructure (Later)

If you want a Boltz-style layout:
- Move kinase-specific code into `hyaline/kinase/`
- Keep `hyaline/loaders/` for shared loaders
- Update `__init__.py` to expose both GPCR and kinase entry points

---

## What Boltz Does

- **Single branch:** `main`
- **Structure:** `src/boltz/` with Boltz-1 and Boltz-2 in the same tree
- **Versions:** Different model configs/weights, not separate repos
- **Relevance:** Same repo, multiple “modes”; they use folders, not branches, for features

---

## Summary Commands

```bash
# 1. Ensure you're on kinase-v1
git checkout kinase-v1

# 2. Verify master is untouched (run from repo root)
git log master -1 --oneline
# Should show: bc6afdd update README

# 3. Rename conflicting package (if doing the fix)
mv hyaline/data hyaline/loaders
# Then run: grep -r "hyaline\.data" --include="*.py" -l . 
# And replace hyaline.data with hyaline.loaders in those files

# 4. Add research/ and key files to commit
git add research/
git add scripts/upload_kinase_to_s3.sh
# etc.

# 5. Update .gitignore
echo "klifs_cache/" >> .gitignore
echo "checkpoints/" >> .gitignore

# 6. Commit
git add .
git status  # Review before committing
git commit -m "feat(kinase): Kinase conformational selectivity + TF modules"

# 7. Push branch
git push -u origin kinase-v1
```

---

## Branch Overview (After Push)

| Branch | Contents | Use Case |
|--------|----------|----------|
| **master** | Original GPCR Hyaline | Production, paper, citations |
| **kinase-v1** | GPCR + Kinase + TF | Development, 6-week sprints |

---

*Last updated: Jan 2026*
