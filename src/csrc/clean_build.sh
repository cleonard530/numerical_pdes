#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: ${SCRIPT_DIR}"
BUILD_DIR="${SCRIPT_DIR}/build"
echo "Build directory: ${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
rm -rf -- *
# After changing CMakeLists.txt (new sources, targets, flags, etc.):
cmake ..
# After only changing .cpp / .h files:
cmake --build .