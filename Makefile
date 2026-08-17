# Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust
# Authors: Ahmad Ali Parr — Jessica Westerhoff
# License: BSL-1.1 / AGPL-3.0 / MPL-2.0
#
# Makefile — PAX GEMM build system for RTX 3080 (Ampere sm_86, 10 GB VRAM)
#
# Usage:
#   make               — build all libraries + test + bench binaries
#   make test          — build + run functional test suite
#   make bench         — build + run all benchmarks
#   make gemm-bench    — build + run standalone GEMM benchmark
#   make inspect-ptx   — dump .ptx for each kernel and grep for key instructions
#   make inspect-sass  — cuobjdump -sass on compiled fatbins
#   make inspect-resources — ptxas resource usage summary
#   make verify        — run tests, print PASS/FAIL
#   make debug         — rebuild with -G -lineinfo (no optimisation)
#   make profile       — run pax_gemm_bench under nvprof
#   make ncu           — run Nsight Compute full profile
#   make clean         — remove all build artefacts

# ═══════════════════════════════════════════════════════════════════════
# TARGET HARDWARE
# ═══════════════════════════════════════════════════════════════════════

ARCH             := sm_86
VRAM_GB          := 10
MAX_PROBLEM_SIZE := 8192

# ═══════════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════════

FUTHARK  ?= futhark
NVCC     ?= nvcc
CC       ?= gcc

# ═══════════════════════════════════════════════════════════════════════
# FLAGS
# ═══════════════════════════════════════════════════════════════════════

CFLAGS   := -O3 -Wall -Wextra -std=c11 -march=native
NVCCFLAGS := -O3 -arch=$(ARCH) -std=c++17 -Xptxas -v

# Additional NVCC flags for device-code line info (added by 'debug' target)
NVCC_DEBUG_FLAGS := -G -lineinfo -O0

# ═══════════════════════════════════════════════════════════════════════
# DIRECTORIES
# ═══════════════════════════════════════════════════════════════════════

BUILD_DIR := build
SRC_DIR   := src

# ═══════════════════════════════════════════════════════════════════════
# ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════

FUTHARK_LIB            := $(BUILD_DIR)/libpax_futhark.so
RTX_VECTOR_LIB         := $(BUILD_DIR)/libpax_rtx_vector.so
RTX_GEMM_WMMA_LIB      := $(BUILD_DIR)/libpax_rtx_gemm_wmma.so
RTX_GEMM_PTX_LIB       := $(BUILD_DIR)/libpax_rtx_gemm_ptx.so
RTX_GEMM_PIPELINE_LIB  := $(BUILD_DIR)/libpax_rtx_gemm_pipeline.so
RTX_GEMM_EPILOGUE_LIB  := $(BUILD_DIR)/libpax_rtx_gemm_epilogue.so
HOST_BRIDGE_OBJ        := $(BUILD_DIR)/host_bridge.o
TEST_BIN               := $(BUILD_DIR)/pax_test
GEMM_BENCH_BIN         := $(BUILD_DIR)/pax_gemm_bench

# PTX output files (for inspection)
PTX_WMMA     := $(BUILD_DIR)/rtx_gemm_wmma.ptx
PTX_PTX      := $(BUILD_DIR)/rtx_gemm_ptx.ptx
PTX_PIPELINE := $(BUILD_DIR)/rtx_gemm_pipeline.ptx
PTX_EPILOGUE := $(BUILD_DIR)/rtx_gemm_epilogue.ptx

ALL_LIBS := $(FUTHARK_LIB) \
            $(RTX_VECTOR_LIB) \
            $(RTX_GEMM_WMMA_LIB) \
            $(RTX_GEMM_PTX_LIB) \
            $(RTX_GEMM_PIPELINE_LIB) \
            $(RTX_GEMM_EPILOGUE_LIB)

# ═══════════════════════════════════════════════════════════════════════
# DEFAULT TARGET
# ═══════════════════════════════════════════════════════════════════════

.PHONY: all
all: $(BUILD_DIR) $(ALL_LIBS) $(HOST_BRIDGE_OBJ) $(TEST_BIN) $(GEMM_BENCH_BIN)
	@echo ""
	@echo "[PAX] Build complete — target $(ARCH), VRAM $(VRAM_GB)GB, max_size $(MAX_PROBLEM_SIZE)"
	@echo "[PAX] Shared memory (pipeline): $$($(BUILD_DIR)/pax_gemm_bench --smem 2>/dev/null || echo 'run pax_gemm_bench to query')"

# ═══════════════════════════════════════════════════════════════════════
# BUILD DIRECTORY
# ═══════════════════════════════════════════════════════════════════════

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ═══════════════════════════════════════════════════════════════════════
# STAGE 1 — Futhark codegen: rtx.fut → C entry points
# ═══════════════════════════════════════════════════════════════════════

$(BUILD_DIR)/rtx_futhark.c $(BUILD_DIR)/rtx_futhark.h: $(SRC_DIR)/rtx.fut | $(BUILD_DIR)
	@echo "[PAX Stage 1] Futhark codegen → CUDA C"
	$(FUTHARK) cuda --library $< -o $(BUILD_DIR)/rtx_futhark

$(FUTHARK_LIB): $(BUILD_DIR)/rtx_futhark.c $(BUILD_DIR)/rtx_futhark.h | $(BUILD_DIR)
	@echo "[PAX Stage 1] Compiling Futhark library → $(FUTHARK_LIB)"
	$(NVCC) $(NVCCFLAGS) --shared --compiler-options '-fPIC' \
		-I$(BUILD_DIR) \
		$(BUILD_DIR)/rtx_futhark.c \
		-o $(FUTHARK_LIB)

# ═══════════════════════════════════════════════════════════════════════
# STAGE 2 — RTX vector kernels (rtx_vector.cu)
# ═══════════════════════════════════════════════════════════════════════

$(RTX_VECTOR_LIB): $(SRC_DIR)/rtx_vector.cu | $(BUILD_DIR)
	@echo "[PAX Stage 2] Compiling RTX vector kernels → $(RTX_VECTOR_LIB)"
	$(NVCC) $(NVCCFLAGS) --shared --compiler-options '-fPIC' \
		$< -o $(RTX_VECTOR_LIB)

# ═══════════════════════════════════════════════════════════════════════
# STAGE 3 — WMMA GEMM (rtx_gemm_wmma.cu)
# ═══════════════════════════════════════════════════════════════════════

$(RTX_GEMM_WMMA_LIB): $(SRC_DIR)/rtx_gemm_wmma.cu | $(BUILD_DIR)
	@echo "[PAX Stage 3] Compiling WMMA GEMM → $(RTX_GEMM_WMMA_LIB)"
	$(NVCC) $(NVCCFLAGS) --shared --compiler-options '-fPIC' \
		--ptxas-options -v \
		$< -o $(RTX_GEMM_WMMA_LIB)

$(PTX_WMMA): $(SRC_DIR)/rtx_gemm_wmma.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --ptx $< -o $(PTX_WMMA)

# ═══════════════════════════════════════════════════════════════════════
# STAGE 4 — PTX inline-assembly GEMM (rtx_gemm_ptx.cu)
# ═══════════════════════════════════════════════════════════════════════

$(RTX_GEMM_PTX_LIB): $(SRC_DIR)/rtx_gemm_ptx.cu | $(BUILD_DIR)
	@echo "[PAX Stage 4] Compiling PTX GEMM → $(RTX_GEMM_PTX_LIB)"
	$(NVCC) $(NVCCFLAGS) --shared --compiler-options '-fPIC' \
		--ptxas-options -v \
		$< -o $(RTX_GEMM_PTX_LIB)

$(PTX_PTX): $(SRC_DIR)/rtx_gemm_ptx.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --ptx $< -o $(PTX_PTX)

# ═══════════════════════════════════════════════════════════════════════
# STAGE 5 — 3-stage pipeline GEMM (rtx_gemm_pipeline.cu)
# ═══════════════════════════════════════════════════════════════════════

$(RTX_GEMM_PIPELINE_LIB): $(SRC_DIR)/rtx_gemm_pipeline.cu | $(BUILD_DIR)
	@echo "[PAX Stage 5] Compiling pipeline GEMM (STAGES=3, sm_86) → $(RTX_GEMM_PIPELINE_LIB)"
	$(NVCC) $(NVCCFLAGS) --shared --compiler-options '-fPIC' \
		--ptxas-options -v \
		$< -o $(RTX_GEMM_PIPELINE_LIB)

$(PTX_PIPELINE): $(SRC_DIR)/rtx_gemm_pipeline.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --ptx $< -o $(PTX_PIPELINE)

# ═══════════════════════════════════════════════════════════════════════
# STAGE 6 — Epilogue fusion GEMM (rtx_gemm_epilogue.cu)
# ═══════════════════════════════════════════════════════════════════════

$(RTX_GEMM_EPILOGUE_LIB): $(SRC_DIR)/rtx_gemm_epilogue.cu | $(BUILD_DIR)
	@echo "[PAX Stage 6] Compiling epilogue GEMM (BiasGeLU / ResidualGeLU / ScaleBiasGeLU) → $(RTX_GEMM_EPILOGUE_LIB)"
	$(NVCC) $(NVCCFLAGS) --shared --compiler-options '-fPIC' \
		--ptxas-options -v \
		$< -o $(RTX_GEMM_EPILOGUE_LIB)

$(PTX_EPILOGUE): $(SRC_DIR)/rtx_gemm_epilogue.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --ptx $< -o $(PTX_EPILOGUE)

# ═══════════════════════════════════════════════════════════════════════
# STAGE 7 — Host bridge C object (host_bridge.c)
# ═══════════════════════════════════════════════════════════════════════

$(HOST_BRIDGE_OBJ): $(SRC_DIR)/host_bridge.c | $(BUILD_DIR)
	@echo "[PAX Stage 7] Compiling host bridge → $(HOST_BRIDGE_OBJ)"
	$(NVCC) $(NVCCFLAGS) -x cu -dc \
		--compiler-options '$(CFLAGS) -fPIC' \
		$< -o $(HOST_BRIDGE_OBJ)

# ═══════════════════════════════════════════════════════════════════════
# STAGE 8 — Test binary (pax_test.c)
# ═══════════════════════════════════════════════════════════════════════

$(TEST_BIN): $(SRC_DIR)/pax_test.c $(HOST_BRIDGE_OBJ) $(ALL_LIBS) | $(BUILD_DIR)
	@echo "[PAX Stage 8] Linking test binary → $(TEST_BIN)"
	$(NVCC) $(NVCCFLAGS) \
		--compiler-options '$(CFLAGS)' \
		$(SRC_DIR)/pax_test.c \
		$(HOST_BRIDGE_OBJ) \
		-L$(BUILD_DIR) \
		-lpax_futhark \
		-lpax_rtx_vector \
		-lpax_rtx_gemm_wmma \
		-lpax_rtx_gemm_ptx \
		-lpax_rtx_gemm_pipeline \
		-lpax_rtx_gemm_epilogue \
		-lcudart -lm \
		-Wl,-rpath,$(BUILD_DIR) \
		-o $(TEST_BIN)

# ═══════════════════════════════════════════════════════════════════════
# STAGE 9 — GEMM benchmark binary (pax_gemm_bench.c)
# ═══════════════════════════════════════════════════════════════════════

$(GEMM_BENCH_BIN): $(SRC_DIR)/pax_gemm_bench.c $(HOST_BRIDGE_OBJ) $(ALL_LIBS) | $(BUILD_DIR)
	@echo "[PAX Stage 9] Linking GEMM benchmark → $(GEMM_BENCH_BIN)"
	$(NVCC) $(NVCCFLAGS) \
		--compiler-options '$(CFLAGS) -DMAX_PROBLEM_SIZE=$(MAX_PROBLEM_SIZE)' \
		$(SRC_DIR)/pax_gemm_bench.c \
		$(HOST_BRIDGE_OBJ) \
		-L$(BUILD_DIR) \
		-lpax_futhark \
		-lpax_rtx_vector \
		-lpax_rtx_gemm_wmma \
		-lpax_rtx_gemm_ptx \
		-lpax_rtx_gemm_pipeline \
		-lpax_rtx_gemm_epilogue \
		-lcudart -lm \
		-Wl,-rpath,$(BUILD_DIR) \
		-o $(GEMM_BENCH_BIN)

# ═══════════════════════════════════════════════════════════════════════
# PHONY TARGETS
# ═══════════════════════════════════════════════════════════════════════

.PHONY: test
test: $(TEST_BIN)
	@echo "[PAX] Running test suite..."
	LD_LIBRARY_PATH=$(BUILD_DIR) ./$(TEST_BIN)

.PHONY: bench
bench: $(GEMM_BENCH_BIN)
	@echo "[PAX] Running all benchmarks..."
	LD_LIBRARY_PATH=$(BUILD_DIR) ./$(GEMM_BENCH_BIN)

.PHONY: gemm-bench
gemm-bench: $(GEMM_BENCH_BIN)
	@echo "[PAX] Running GEMM benchmark (m=n=k=$(MAX_PROBLEM_SIZE))..."
	LD_LIBRARY_PATH=$(BUILD_DIR) ./$(GEMM_BENCH_BIN) \
		--m $(MAX_PROBLEM_SIZE) --n $(MAX_PROBLEM_SIZE) --k $(MAX_PROBLEM_SIZE)

# -----------------------------------------------------------------------
# PTX / SASS inspection targets
# -----------------------------------------------------------------------

.PHONY: inspect-ptx
inspect-ptx: $(PTX_WMMA) $(PTX_PTX) $(PTX_PIPELINE) $(PTX_EPILOGUE)
	@echo ""
	@echo "[PAX] ── PTX instruction audit ──────────────────────────────"
	@for f in $(PTX_WMMA) $(PTX_PTX) $(PTX_PIPELINE) $(PTX_EPILOGUE); do \
	    echo ""; \
	    echo "  $$f:"; \
	    echo "    mma.sync     : $$(grep -c 'mma\.sync'    $$f || echo 0)"; \
	    echo "    ldmatrix     : $$(grep -c 'ldmatrix'     $$f || echo 0)"; \
	    echo "    cp.async     : $$(grep -c 'cp\.async'    $$f || echo 0)"; \
	done
	@echo ""

.PHONY: inspect-sass
inspect-sass: $(RTX_GEMM_WMMA_LIB) $(RTX_GEMM_PTX_LIB) \
              $(RTX_GEMM_PIPELINE_LIB) $(RTX_GEMM_EPILOGUE_LIB)
	@echo ""
	@echo "[PAX] ── SASS instruction audit ─────────────────────────────"
	@for f in $^; do \
	    echo ""; \
	    echo "  $$f:"; \
	    cuobjdump -sass $$f 2>/dev/null | \
	        grep -E 'HMMA|LDMATRIX|CP\.ASYNC|STS|STG' | \
	        awk '{counts[$$2]++} END {for (k in counts) print "    " k ": " counts[k]}' | \
	        sort -t: -k2 -rn; \
	done
	@echo ""

.PHONY: inspect-resources
inspect-resources: $(PTX_WMMA) $(PTX_PTX) $(PTX_PIPELINE) $(PTX_EPILOGUE)
	@echo ""
	@echo "[PAX] ── ptxas resource usage ───────────────────────────────"
	@for f in $(PTX_WMMA) $(PTX_PTX) $(PTX_PIPELINE) $(PTX_EPILOGUE); do \
	    echo ""; \
	    echo "  $$f:"; \
	    ptxas -arch $(ARCH) -v $$f -o /dev/null 2>&1 | \
	        grep -E 'registers|lmem|smem|cmem|ptxas info'; \
	done
	@echo ""

.PHONY: verify
verify: test
	@echo "[PAX] Verification complete."

.PHONY: debug
debug:
	@echo "[PAX] Rebuilding in debug mode (-G -lineinfo)..."
	$(MAKE) clean
	$(MAKE) all NVCCFLAGS="$(NVCC_DEBUG_FLAGS) -arch=$(ARCH) -std=c++17"

.PHONY: profile
profile: $(GEMM_BENCH_BIN)
	@echo "[PAX] nvprof profile..."
	LD_LIBRARY_PATH=$(BUILD_DIR) \
	nvprof --metrics achieved_occupancy,sm_efficiency,dram_read_throughput,dram_write_throughput \
	    ./$(GEMM_BENCH_BIN)

.PHONY: ncu
ncu: $(GEMM_BENCH_BIN)
	@echo "[PAX] Nsight Compute full profile → $(BUILD_DIR)/pax_gemm_ncu.ncu-rep"
	LD_LIBRARY_PATH=$(BUILD_DIR) \
	ncu --target-processes all --set full \
	    -o $(BUILD_DIR)/pax_gemm_ncu \
	    ./$(GEMM_BENCH_BIN)
	@echo "[PAX] Report written to $(BUILD_DIR)/pax_gemm_ncu.ncu-rep"

.PHONY: clean
clean:
	@echo "[PAX] Cleaning build artefacts..."
	rm -rf $(BUILD_DIR)
	@echo "[PAX] Clean complete."
