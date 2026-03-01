#!/bin/bash
set -e

echo "Building WASM..."
wasm-pack build --target web --out-dir www/pkg

echo "Compiling TypeScript..."
tsc -p www/tsconfig.json

# Sync HDR assets from assets/ → www/assets/ so the HTTP server can serve them.
# cp -n skips files that already exist to avoid re-copying large binaries needlessly.
echo "Syncing assets..."
mkdir -p www/assets
cp -n assets/*.hdr www/assets/ 2>/dev/null || true

echo "Build complete."
