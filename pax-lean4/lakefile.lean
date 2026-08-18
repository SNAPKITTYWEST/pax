-- PAX GEMM Verification — Lake Build Configuration
-- Requires: Lean 4.11+, mathlib4
-- Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust
-- Authors: Ahmad Ali Parr — Jessica Westerhoff

import Lake
open Lake DSL

require mathlib from git
  "https://github.com/leanprover/mathlib4" @ "v4.11.0"

@[default_target]
lean_lib PAX where
  srcDir := "."
  globs := #[.andSubmodules `PAX]

lean_exe pax_extract where
  root := `PAX.Extraction
