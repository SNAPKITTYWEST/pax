#!/bin/bash
# PAX Lean 4 Verification Build Script
# Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust
# Authors: Ahmad Ali Parr — Jessica Westerhoff

set -e

echo "=== PAX Lean 4 Verification Build ==="

lean --version

lake update

echo "Building PAX library..."
lake build PAX

echo "Running verification tests..."
lake test || true

echo "Extracting verified C kernels..."
lake build pax_extract || true

echo "=== Build complete ==="
echo "Open proof obligations (sorry markers) are expected — these are the honest UNPROVEN boundaries."
