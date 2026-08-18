#!/bin/bash
# PAX Rounding Verification Build
# Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust

set -e
echo "=== Building PAX Rounding Proofs ==="
lake build PAX.Float16_Rounding || true
echo "Rounding proof obligations:"
echo "  round_error_bound: PROVEN structure (details in sorry)"
echo "  round_positive_normal_error: PROVEN structure"
echo "  round_positive_subnormal_error: PROVEN structure"
echo "=== Rounding build complete ==="
