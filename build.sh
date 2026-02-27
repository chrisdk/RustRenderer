#!/bin/bash
set -e

echo "Building WASM..."
wasm-pack build --target web --out-dir www/pkg

echo "Compiling TypeScript..."
tsc -p www/tsconfig.json

echo "Build complete."
