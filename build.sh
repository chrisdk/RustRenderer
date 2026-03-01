#!/bin/bash
set -e

echo "Building WASM..."
wasm-pack build --target web --out-dir www/pkg

echo "Compiling TypeScript..."
tsc -p www/tsconfig.json

# Inject build version. We use the total git commit count as a monotonically
# increasing build number. The script sets the text of #build-version in the
# DOM so index.html never needs to be modified by the build process.
echo "Injecting build version..."
# +1 because build.sh runs before the publish commit is made, so the count
# at this moment is N-1 relative to the commit that will contain these files.
# Adding 1 makes the displayed number match the commit it ships in.
BUILD=$(( $(git rev-list --count HEAD 2>/dev/null || echo "0") + 1 ))
cat > www/dist/version.js << EOF
(function () {
    var el = document.getElementById('build-version');
    if (el) el.textContent = 'build ${BUILD}';
})();
EOF

# Sync HDR assets from assets/ → www/assets/ so the HTTP server can serve them.
# cp -n skips files that already exist to avoid re-copying large binaries needlessly.
echo "Syncing assets..."
mkdir -p www/assets
cp -n assets/*.hdr www/assets/ 2>/dev/null || true

echo "Build complete."
