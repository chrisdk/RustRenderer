#!/bin/bash
set -e

./build.sh

echo "Serving at http://localhost:8080"
python3 -m http.server 8080 --directory www
