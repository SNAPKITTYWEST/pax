#!/bin/bash
# PAX Float16 Formalization Build
# Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust

set -e
echo "=== Building PAX Float16 Formalization ==="
lake build PAX.Float16
lake build PAX.Float32
lake build PAX.Float16_Rounding || true
echo "=== Float16 build complete ==="
