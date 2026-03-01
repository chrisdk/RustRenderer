#!/bin/bash
#
# publish.sh — build, stage, commit, and push everything to GitHub.
#
# Usage:
#   ./publish.sh                        # commit message defaults to "Publish"
#   ./publish.sh "Add FOV slider"       # custom commit message
#
# The script stages all tracked source files plus the compiled WASM and JS
# artifacts. node_modules, /target, and other large generated directories
# are excluded by .gitignore and are never touched.

set -e

MSG="${1:-Publish}"

echo "==> Building..."
./build.sh

# wasm-pack regenerates www/pkg/.gitignore on every build, which ignores
# every file in that directory. Remove it so the compiled output is visible
# to git and can be committed for GitHub Pages.
rm -f www/pkg/.gitignore

echo "==> Staging..."
git add \
    src/ \
    www/src/ www/index.html www/pkg/ www/dist/ www/assets/ \
    BACKLOG.md \
    Cargo.toml Cargo.lock \
    build.sh run.sh publish.sh \
    .gitignore

# Also pick up any other tracked files that have been modified (e.g. README,
# CLAUDE.md, LICENSE) without accidentally sweeping in untracked junk.
git add -u

if git diff --cached --quiet; then
    echo "==> Nothing changed — already up to date."
    exit 0
fi

echo "==> Committing: \"$MSG\""
git commit -m "$MSG

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

echo "==> Pushing to GitHub..."
git push

echo "==> Done. Live at https://chrisdk.github.io/RustRenderer/"
