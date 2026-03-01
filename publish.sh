#!/bin/bash
#
# publish.sh — build, assemble docs/, commit, and push to GitHub Pages.
#
# Usage:
#   ./publish.sh                        # commit message defaults to "Publish"
#   ./publish.sh "Add FOV slider"       # custom commit message
#
# www/ is the development workspace. This script copies only the files the
# browser needs (index.html, compiled WASM, compiled JS, HDR assets) into
# docs/, which is what GitHub Pages serves. Source files and build
# intermediates never end up in docs/.

set -e

MSG="${1:-Publish}"

echo "==> Building..."
./build.sh

echo "==> Assembling docs/..."
rm -rf docs
mkdir -p docs
cp    www/index.html docs/
cp -r www/pkg        docs/
cp -r www/dist       docs/
cp -r www/assets     docs/

# wasm-pack generates a .gitignore inside pkg/ that ignores everything in
# that directory. Remove it so the compiled WASM is visible to git.
rm -f docs/pkg/.gitignore

echo "==> Staging..."
# Stage the freshly assembled deployment directory.
git add docs/
# Stage any source-level changes (Rust, TypeScript, HTML, scripts, etc).
git add \
    src/ \
    www/src/ www/index.html www/assets/ \
    BACKLOG.md \
    Cargo.toml Cargo.lock \
    build.sh run.sh publish.sh \
    .gitignore
# Pick up modifications to any other already-tracked files (README, CLAUDE.md…).
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
