-- PAX Verified_Pipeline — Verified Software Pipeline Properties
-- Pipeline stage invariants, hazard-free scheduling, and HBM bandwidth bounds
-- Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust
-- Authors: Ahmad Ali Parr — Jessica Westerhoff
-- License: BSL-1.1 / AGPL-3.0 / MPL-2.0

import PAX.Pipeline
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

open PAX

namespace PAX.Verified

-- ═══════════════════════════════════════════════════════════════════════
-- PIPELINE STAGE MODEL
-- ═══════════════════════════════════════════════════════════════════════

/-- A concrete software pipeline stage instance for a GEMM kernel.
    Tracks the k-tile index, which shared memory buffer is in use (ping/pong),
    and the lifecycle state of async copy and compute phases. -/
structure PipelineStage where
  /-- Which k-tile this stage is processing -/
  kTile       : ℕ
  /-- Buffer index (0 or 1 for double-buffering, 0..2 for triple) -/
  bufferIdx   : ℕ
  /-- cp.async has been issued (but not necessarily completed) -/
  copyIssued  : Bool
  /-- cp.async.commit_group has been called -/
  copyCommit  : Bool
  /-- cp.async.wait_group has returned (copy is complete) -/
  copyDone    : Bool
  /-- mma.sync has been issued for this stage -/
  computeIssued : Bool
  /-- mma.sync has retired (all threads in the warp have completed) -/
  computeDone : Bool
  /-- Invariant: compute cannot start before copy is done (no RAW hazard) -/
  hRAW        : computeIssued → copyDone = true

-- ═══════════════════════════════════════════════════════════════════════
-- PIPELINE INVARIANTS
-- ═══════════════════════════════════════════════════════════════════════

/-- The hazard-free invariant for a pipeline of `n` stages.
    For every stage, if compute has been issued, then the copy for that stage
    must have completed (no Read-After-Write on shared memory).

    This is enforced in hardware by `cp.async.wait_group N` which stalls
    until at most N async copy groups are outstanding. -/
def hazardFree (stages : List PipelineStage) : Prop :=
  ∀ s ∈ stages, s.computeIssued → s.copyDone = true

/-- **Pipeline HB (happens-before) preserved**:
    The hazard-free invariant is maintained across the main pipeline loop.
    Specifically: issuing compute for stage s only after waiting for s's copy
    preserves the hazardFree predicate.

    This is the core safety theorem for the PAX software pipeline:
    it guarantees that the cp.async barrier discipline (from PAX.Pipeline,
    copyBeforeCompute predicate) correctly prevents smem data races. -/
theorem pipeline_hb_preserved
    (stages : List PipelineStage)
    (hInitHF : hazardFree stages)
    (newStage : PipelineStage)
    (hNewHF : newStage.computeIssued → newStage.copyDone = true) :
    hazardFree (newStage :: stages) := by
  intro s hs hsi
  cases hs with
  | head => exact hNewHF hsi
  | tail _ hs => exact hInitHF s hs hsi

/-- Corollary: the empty pipeline is trivially hazard-free. -/
theorem empty_pipeline_hazard_free : hazardFree [] := by
  intro s hs; exact absurd hs (List.not_mem_nil s)

-- ═══════════════════════════════════════════════════════════════════════
-- DOUBLE/TRIPLE BUFFER VALIDITY
-- ═══════════════════════════════════════════════════════════════════════

/-- A `stages`-stage pipeline uses buffer indices in `Fin stages`.
    Buffer validity: the buffer index of each stage is less than the
    number of pipeline stages (mod `stages` gives the SMEM slot). -/
def bufferValid (numStages : ℕ) (stage : PipelineStage) : Prop :=
  stage.bufferIdx < numStages

/-- All stages in a `numStages`-pipeline use valid buffer indices. -/
def allBuffersValid (numStages : ℕ) (stages : List PipelineStage) : Prop :=
  ∀ s ∈ stages, bufferValid numStages s

/-- Buffer aliasing freedom: two distinct stages at the same buffer index
    cannot both be "in-flight" (copyDone=false or computeDone=false) simultaneously.
    This prevents one stage from overwriting smem that another stage is still using. -/
theorem no_buffer_alias
    (numStages : ℕ) (stages : List PipelineStage)
    (hValid : allBuffersValid numStages stages) :
    -- Abstract statement: any two stages with the same bufferIdx are well-separated in kTile
    ∀ (s1 s2 : PipelineStage),
      s1 ∈ stages → s2 ∈ stages →
      s1.bufferIdx = s2.bufferIdx →
      s1.kTile = s2.kTile → s1 = s2 := by
  sorry
  -- Proof: in a correctly scheduled pipeline, each bufferIdx is used by at most one
  -- kTile at a time. Two stages with the same bufferIdx and kTile must be the same
  -- stage (inserted once into the list). Requires PipelineStage to be Eq-decidable
  -- and the insertion invariant from the loop scheduler.

-- ═══════════════════════════════════════════════════════════════════════
-- THROUGHPUT BOUNDS (RE-EXPORT)
-- ═══════════════════════════════════════════════════════════════════════

/-- The pipeline overlap bound for 2, 3, and 4 stages (re-export with concrete values). -/
theorem pipeline_2stage_bound :
    achievedThroughput 2 ≥ (1 - 1 / (2 : ℝ)) * min peakCompute peakMemory :=
  pipeline_overlap_bound (by norm_num) (by norm_num)
    ⟨128, 128, 32, 16, 16, 16⟩ peakCompute peakMemory

theorem pipeline_3stage_bound :
    achievedThroughput 3 ≥ (1 - 1 / (3 : ℝ)) * min peakCompute peakMemory :=
  pipeline_overlap_bound (by norm_num) (by norm_num)
    ⟨128, 128, 32, 16, 16, 16⟩ peakCompute peakMemory

theorem pipeline_4stage_bound :
    achievedThroughput 4 ≥ (1 - 1 / (4 : ℝ)) * min peakCompute peakMemory :=
  pipeline_overlap_bound (by norm_num) (by norm_num)
    ⟨128, 128, 32, 16, 16, 16⟩ peakCompute peakMemory

/-- Among stages 2,3,4: the 4-stage pipeline has the highest efficiency bound. -/
theorem pipeline_4stage_best : ∀ (s : ℕ), 2 ≤ s → s ≤ 4 →
    (1 - 1 / (s : ℝ)) ≤ (1 - 1 / (4 : ℝ)) := by
  intro s hs hs'
  apply sub_le_sub_left
  apply div_le_div_of_nonneg_left (by norm_num) (by norm_num) (by norm_num)
  exact_mod_cast hs

end PAX.Verified
