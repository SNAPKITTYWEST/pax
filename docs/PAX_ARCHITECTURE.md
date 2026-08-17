# PAX: Sovereign GPU Computing Model
## Parallel Accelerator eXecution Architecture
### A First-Principles Reconstruction of Data-Parallel Computation

---

## 0. Foundational Principles

Before any layer is constructed, we establish five axioms that govern the entire stack.
These are not derived from CUDA, OpenCL, or any existing proprietary model. They are
derived from the physics of parallel execution and the mathematics of concurrent state
transition.

**Axiom 1 — Index Space Primacy**: Parallelism is a geometric property of an index
space, not a hardware property. A kernel is a function over a discrete topological space.
Hardware merely provides a mapping from this space to physical execution units.

**Axiom 2 — Permission Necessity**: Memory safety in a parallel context cannot be
guaranteed by type systems alone. It requires an explicit permission algebra. Every
memory access must be justified by a held permission.

**Axiom 3 — Synchronization as State Transition**: A synchronization primitive is not a
"pause." It is a well-defined state transition on the global configuration of permissions
and memory visibility. Barriers redistribute permissions.

**Axiom 4 — Warp Distinctness**: The warp (or wavefront) is the atomic unit of
scheduling. The thread is the atomic unit of semantics. Confusing these leads to
incorrect divergence handling and invalid barrier placement.

**Axiom 5 — Verification Non-Negotiability**: No component is considered valid unless it
has associated proof obligations. Testing demonstrates existence of bugs; proofs
demonstrate absence of classes of bugs.

---

## LAYER 0 — MATHEMATICAL MODEL

### 0.1 Index Spaces and Kernel Functions

**[MATHEMATICAL FOUNDATION]**

Let ℤ⁺ denote the non-negative integers. An index space is a finite subset of a
d-dimensional integer lattice:

```
I = [0, G_x) × [0, G_y) × [0, G_z) ⊂ ℤ⁺³
```

where d ∈ {1, 2, 3}. Each point t ∈ I is a thread index. A kernel is a function:

```
K: I × M → M
```

where M is the set of all valid memory states. The kernel is data-parallel when K(t, m)
depends only on t and a subset of m determined by t.

A work-group (or thread block) is a sub-lattice partition of I. Given block dimensions
(B_x, B_y, B_z), the work-group space is:

```
G = [0, ⌈G_x/B_x⌉) × [0, ⌈G_y/B_y⌉) × [0, ⌈G_z/B_z⌉)
```

Each work-group g ∈ G contains thread indices:

```
T_g = { t ∈ I | ⌊t_i / B_i⌋ = g_i for all i ∈ {x,y,z} }
```

The local index of a thread within its work-group is l = t mod B.

**[ARCHITECTURE]**

This model directly corresponds to the OpenCL NDRange and CUDA grid/block
abstractions, but derived without reference to either. The index space is the primary
object; hardware mapping is secondary. The mathematical structure is a fiber bundle:
the global index space I is the total space, work-groups are fibers, and the projection
π: I → G maps threads to their containing work-group.

**[IMPLEMENTATION PLAN]**

- Represent index spaces as tuples of ranges in the compiler frontend
- Encode index space dimensions as compile-time constants where possible
- Runtime stores grid and block dimensions in launch descriptors
- CPU simulation iterates over the index space with nested loops

**[FORMAL MODEL]**

```lean4
structure IndexSpace where
  dims : Fin 3 → Nat
  def contains (i : Fin 3 → Nat) : Prop :=
    ∀ d, i d < dims d

structure Kernel (α : Type) where
  indexSpace : IndexSpace
  body : (Fin 3 → Nat) → StateM Memory α

def workGroupOf (grid block : IndexSpace) (t : Fin 3 → Nat) : Fin 3 → Nat :=
  fun d => t d / block.dims d

def localIdOf (block : IndexSpace) (t : Fin 3 → Nat) : Fin 3 → Nat :=
  fun d => t d % block.dims d
```

**[CONSTRAINTS]**

- ∀ i ∈ {x,y,z}: B_i > 0 (block dimensions strictly positive)
- ∀ i ∈ {x,y,z}: G_i > 0 (grid dimensions strictly positive)
- |T_g| = B_x × B_y × B_z for all interior work-groups (boundary work-groups may be
  smaller if grid does not divide evenly)
- Work-groups are disjoint: g ≠ g' ⟹ T_g ∩ T_{g'} = ∅

**[TEST STRATEGY]**

- Property: ⋃_{g ∈ G} T_g = I (partition coverage)
- Property: ∀ g, g' ∈ G, g ≠ g' ⟹ T_g ∩ T_{g'} = ∅ (partition disjointness)
- Property: ∀ t ∈ I, workGroupOf(grid, block, t) ∈ G (valid mapping)
- Fuzz test with random grid/block dimensions

**[NEXT EXPERIMENT]**

Generalize to non-rectangular index spaces (e.g., triangular matrices, sparse index
sets) using affine constraints from the polyhedral model.

---

### 0.2 Tensors and Memory as Partial Functions

**[MATHEMATICAL FOUNDATION]**

A tensor of rank r with shape S = (s₁, …, s_r) and element type τ is a partial function:

```
T: [0, s₁) × ··· × [0, s_r) ⇀ τ
```

where ⇀ denotes a partial function. The domain of definition dom(T) represents
initialized elements.

Memory is a global partial function:

```
M: Addr ⇀ Value
```

where Addr is a set of physical or virtual addresses.

A memory operation is one of:
- Read(a): M → M × Value where a ∈ dom(M)
- Write(a, v): M → M' where M' = M[a ↦ v]
- Atomic(a, f): M → M' × Value where M' = M[a ↦ f(M(a))]

Address spaces are disjoint subsets of Addr:

```
Addr = Addr_private ⊎ Addr_local ⊎ Addr_global ⊎ Addr_constant
```

**[FORMAL MODEL]**

```lean4
inductive AddrSpace
  | private | local | global | constant

def Memory (α : Type) := AddrSpace → Nat → Option α

def read (M : Memory α) (as : AddrSpace) (a : Nat) : Option α := M as a

def write (M : Memory α) (as : AddrSpace) (a : Nat) (v : α) : Memory α :=
  fun as' a' => if as = as' ∧ a = a' then some v else M as' a'

theorem read_after_write (M : Memory α) (as : AddrSpace) (a : Nat) (v : α) :
  read (write M as a v) as a = some v := by simp [read, write]
```

**[CONSTRAINTS]**

- Addr_private^(t) ∩ Addr_private^(t') = ∅ for t ≠ t' (private memory is per-thread)
- Addr_local^(g) ∩ Addr_local^(g') = ∅ for g ≠ g' (local memory is per-work-group)
- Write to constant address space is forbidden
- Atomic operations require aligned addresses

---

### 0.3 SIMT Execution and Warp Partition

**[MATHEMATICAL FOUNDATION]**

The Single Instruction Multiple Thread (SIMT) model arises from hardware constraints:
it is more area-efficient to share instruction decode and control logic among multiple
execution lanes. Given a warp width W ∈ {32, 64}, the warp partition is:

```
P_W(I) = { W_j | j ∈ J }
```

where each W_j contains exactly W thread indices, and ⋃_j W_j = I with W_j ∩ W_{j'} = ∅.

Within a warp, threads execute in lockstep: at any cycle, all active lanes execute the
same instruction. The active mask (EXEC mask) is:

```
EXEC: W_j → {0, 1}
```

Divergence is resolved by a reconvergence stack:

1. At a conditional branch, push the not-taken mask onto the stack
2. Execute the taken path with the active mask
3. When the path ends, pop the stack and execute the not-taken path
4. Reconverge when both paths complete

**[FORMAL MODEL]**

```lean4
structure Warp (W : Nat) where
  threads : Fin W → ThreadId
  execMask : Fin W → Bool

inductive SIMTState
  | executing (pc : Nat) (exec : Warp W)
  | diverged (stack : List (Warp W)) (current : Warp W)

def stepSIMT (s : SIMTState) (inst : Instruction) : SIMTState :=
  match s with
  | executing pc exec =>
      if inst.isBranch then
        let (taken, notTaken) := evalBranch exec inst
        if notTaken.isEmpty then executing (pc + 1) taken
        else diverged [notTaken] taken
      else executing (pc + 1) (executeAll exec inst)
  | diverged (top :: rest) current =>
      if current.isEmpty then
        if rest.isEmpty then executing top.pc top
        else diverged rest top
      else diverged (top :: rest) (step current)
```

**[CONSTRAINTS]**

- Divergence within a warp at a barrier is undefined behavior
- Reconvergence must occur before the next barrier or warp exit
- A barrier may only be reached by all active threads in a work-group or none

---

### 0.4 Memory Model and Happens-Before

**[MATHEMATICAL FOUNDATION]**

Let ℰ be the set of memory events. The happens-before relation HB ⊆ ℰ × ℰ is a
strict partial order satisfying:

1. Program order: e₁ before e₂ in same thread ⟹ (e₁, e₂) ∈ HB
2. Synchronization order: release e₁ visible to acquire e₂ ⟹ (e₁, e₂) ∈ HB
3. Transitivity

A data race:

```
DataRace(e₁, e₂) ⟺ addr(e₁) = addr(e₂)
                   ∧ (isWrite(e₁) ∨ isWrite(e₂))
                   ∧ (e₁, e₂) ∉ HB
                   ∧ (e₂, e₁) ∉ HB
```

**[FORMAL MODEL]**

```lean4
inductive HappensBefore : Event → Event → Prop
  | po  : ∀ e1 e2, sameThread e1 e2 → programOrder e1 e2 → HappensBefore e1 e2
  | sync : ∀ e1 e2, releaseAcquire e1 e2 → HappensBefore e1 e2
  | trans : ∀ e1 e2 e3, HappensBefore e1 e2 → HappensBefore e2 e3 → HappensBefore e1 e3

def DataRace (e1 e2 : Event) : Prop :=
  e1.addr = e2.addr ∧ (e1.kind = write ∨ e2.kind = write) ∧
  ¬(HappensBefore e1 e2) ∧ ¬(HappensBefore e2 e1)
```

**[CONSTRAINTS]**

- All atomic operations are sequentially consistent by default
- Barrier operations imply a fence on all address spaces
- No data races in valid programs (enforced by verifier)

---

### 0.5 Permission Algebra and Separation Logic

**[MATHEMATICAL FOUNDATION]**

A permission p ∈ Perm is a value in a permission algebra (Perm, ⊕, 0) where:
- ⊕ is a partial binary operation (composition)
- 0 is the empty permission (unit)

For memory, fractional permissions:
- Perm = [0, 1] where 1 is full write permission and (0, 1] is read permission
- p₁ ⊕ p₂ is defined iff p₁ + p₂ ≤ 1
- Write requires permission 1; read requires permission > 0

The separating conjunction P * Q holds for state σ iff there exist σ₁, σ₂ such that
σ = σ₁ ⊎ σ₂, P holds for σ₁, and Q holds for σ₂.

**[FORMAL MODEL]**

```lean4
def Perm := { r : Rat // 0 ≤ r ∧ r ≤ 1 }

def compose (p1 p2 : Perm) : Option Perm :=
  let sum := p1.val + p2.val
  if sum ≤ 1 then some ⟨sum, by sorry⟩ else none

def writePerm : Perm := ⟨1, by norm_num⟩

structure PermissionHeap where
  permissions : Nat → Option Perm
```

**[CONSTRAINTS]**

- ∀ a, ∑_{t ∈ Threads} p_t(a) ≤ 1 (no permission overflow)
- A thread may only access address a if it holds permission > 0 for a
- Barrier redistribution must preserve total permission: ∑p_before = ∑p_after

---

## LAYER 1 — ABSTRACT GPU MACHINE

### 1.1 Compute Unit Architecture

**[MATHEMATICAL FOUNDATION]**

A Compute Unit (CU) is a tuple C = (L, R, A, M_local) where:
- L = {0, …, N_L−1} is the set of execution lanes
- R = L × RegId → Value is the register file
- A is the set of ALUs, partitioned into scalar and vector
- M_local is the local memory (shared among all lanes)

**[ARCHITECTURE]**

From AMD GCN/CDNA and NVIDIA SM architectures:
- Each CU has N_L lanes (typically 32 or 64)
- A scalar unit handles uniform operations (same value for all lanes)
- A vector unit handles divergent operations (per-lane values)
- Local memory (LDS on AMD, shared memory on NVIDIA) is SRAM

**[FORMAL MODEL]**

```lean4
structure ComputeUnit (NL : Nat) where
  lanes    : Fin NL → LaneState
  regFile  : Fin NL → RegId → Option Value
  scalarUnit : ScalarState
  localMem : Nat → Option Value

inductive CUInstruction
  | scalar (op : ScalarOp) (dst : RegId) (srcs : List RegId)
  | vector (op : VectorOp) (dst : RegId) (srcs : List RegId)
  | localLoad  (dst : RegId) (addr : RegId)
  | localStore (src : RegId) (addr : RegId)
  | barrier
```

**[CONSTRAINTS]**

- 0 < N_L ≤ 64 (lane count bounded by hardware)
- Register file size is finite; spilling to local memory required if exceeded
- Local memory size is finite per work-group (typically 32–64 KB)

---

### 1.2 Warp/Wavefront Model

**[FORMAL MODEL]**

```lean4
structure WarpState (W : Nat) where
  pc       : Nat
  execMask : Fin W → Bool
  regFile  : Fin W → RegId → Option Value
  divStack : List (Fin W → Bool)

def warpStep (warp : WarpState W) (inst : Instruction) : WarpState W :=
  if inst.isBranch then
    let conds       := evalConditions warp inst
    let takenMask   := fun i => warp.execMask i && conds i
    let notTakenMask := fun i => warp.execMask i && !conds i
    if (∀ i, notTakenMask i = false) then
      { warp with pc := inst.takenTarget }
    else
      { warp with pc := inst.takenTarget,
                  execMask := takenMask,
                  divStack := notTakenMask :: warp.divStack }
  else
    { warp with pc := warp.pc + 1,
                regFile := executeInstruction warp inst }
```

**[CONSTRAINTS]**

- Warps within a work-group execute on the same CU
- Warp size W is a power of two (32 or 64)
- A warp may not span work-group boundaries

---

### 1.3 Memory Hierarchy

| Region   | Scope           | Typical Latency | Typical Capacity |
|----------|-----------------|-----------------|------------------|
| Private  | Single thread   | 1 cycle         | 256 KB (regs)    |
| Local    | Work-group      | 10–20 cycles    | 32–64 KB         |
| Global   | All threads     | 100–400 cycles  | GBs              |
| Constant | All (read-only) | 10–20 cycles    | 64 KB            |
| Host     | CPU + GPU       | 1000+ cycles    | System RAM       |

**[FORMAL MODEL]**

```lean4
def scopeOf : MemRegion → Set ThreadId
  | private  => fun t => {t}
  | local    => fun t => workGroupOf t
  | global   => fun _ => Set.univ
  | constant => fun _ => Set.univ
  | host     => fun _ => Set.univ

def latencyOf : MemRegion → Nat
  | private  => 1
  | local    => 15
  | global   => 200
  | constant => 15
  | host     => 2000
```

---

### 1.4 Scheduling Model

**[MATHEMATICAL FOUNDATION]**

The scheduler is a function:

```
Schedule: W × State → W × State
```

Scheduling policies:
- Round-robin: next(w) = (w + 1) mod |W|
- Greedy: select the warp with the oldest instruction
- Priority: select based on warp priority bits

A Time-Slice Group (TSG) is a set of warps that are preempted together.

**[FORMAL MODEL]**

```lean4
structure SchedulerState where
  warps       : List (WarpState W)
  readyQueue  : List Nat
  waitingQueue : List (Nat × WaitReason)

inductive WaitReason
  | memory   (addr : Nat)
  | barrier  (workGroup : Nat)
  | dependency (event : EventId)

def schedule (s : SchedulerState) : SchedulerState × Nat :=
  match s.readyQueue with
  | []      => (s, 0)
  | w :: ws => ({ s with readyQueue := ws }, w)
```

**[CONSTRAINTS]**

- At most one warp issues instructions per cycle per CU
- Warps waiting at barriers cannot be scheduled until all warps in the work-group
  reach the barrier
- TSG save/restore must preserve all architectural state

---

### 1.5 Instruction Set Architecture

**[FORMAL MODEL]**

```lean4
inductive PAXInstruction
  | scalarAdd  (dst : RegId) (src1 src2 : RegId)
  | vectorFMA  (dst : RegId) (src1 src2 src3 : RegId)
  | load       (dst : RegId) (addr : RegId) (as : AddrSpace)
  | store      (src : RegId) (addr : RegId) (as : AddrSpace)
  | atomicAdd  (dst : RegId) (addr : RegId) (src : RegId)
  | branch     (cond : RegId) (target : Nat)
  | barrier    (scope : BarrierScope)
  | warpReduce (op : ReduceOp) (dst : RegId) (src : RegId)
  | ret

def latency : PAXInstruction → Nat
  | scalarAdd  _ _ _     => 1
  | vectorFMA  _ _ _ _   => 4
  | load       _ _ as    => latencyOf as
  | store      _ _ as    => latencyOf as
  | atomicAdd  _ _ _     => 20
  | branch     _ _       => 1
  | barrier    _         => 100
  | warpReduce _ _ _     => 4
  | ret                  => 1
```

**[CONSTRAINTS]**

- Scalar instructions execute on the scalar unit, not the vector ALUs
- Barrier instructions must be reached by all active threads in scope
- Collective operations require converged warps (no divergence)

---

## LAYER 2 — PAX INTERMEDIATE REPRESENTATION

### 2.1 IR Structure and Design Philosophy

PAX-IR is a structured, SSA-based intermediate representation. A module is a directed
acyclic graph of functions. A function is a control-flow graph of basic blocks. A basic
block is a sequence of operations.

Every parallel construct is represented as an operation with well-defined semantics.
The index space is parameterized, not assumed.

**[FORMAL MODEL]**

```lean4
structure PAXModule where
  functions : List PAXFunction

structure PAXFunction where
  name      : String
  arguments : List (String × PAXType)
  blocks    : List PAXBlock

structure PAXBlock where
  label      : String
  operations : List PAXOperation
  terminator : PAXTerminator

inductive PAXOperation
  | launch    (grid block : IndexSpace) (body : PAXFunction)
  | threadId  (dim : Fin 3)
  | blockId   (dim : Fin 3)
  | load      (ptr : Value) (as : AddrSpace)
  | store     (ptr val : Value) (as : AddrSpace)
  | barrier   (scope : BarrierScope)
  | atomic    (op : AtomicOp) (ptr val : Value) (as : AddrSpace)
  | reduce    (op : ReduceOp) (val : Value) (scope : BarrierScope)
  | alloca    (size : Nat) (as : AddrSpace)
  | arith     (op : ArithOp) (operands : List Value)
```

**[CONSTRAINTS]**

- All values are in SSA form
- Basic blocks must have exactly one terminator
- `launch` operations may only appear at the top level or in host functions
- `barrier` operations may only appear inside launch bodies
- All memory operations must specify an address space

---

### 2.2 Core Operation Semantics

**[MATHEMATICAL FOUNDATION]**

`pax.launch`:
```
⟦pax.launch G, B, f⟧(σ) = ∏_{t ∈ G × B} ⟦f(t)⟧(σ)
```
(∏ denotes parallel composition)

`pax.barrier`:
```
⟦pax.barrier s⟧(σ) = sync_s(σ)
```

`pax.atomic`:
```
⟦pax.atomic op, ptr, val⟧(σ) = (σ[M(ptr) ↦ op(σ.M(ptr), val)], σ.M(ptr))
```

**[FORMAL MODEL]**

```lean4
def evalOp (op : PAXOperation) (state : PAXState) : PAXState × Option Value :=
  match op with
  | launch grid block body =>
      let newState := parallelFor grid block (fun t => evalFunction body t) state
      (newState, none)
  | barrier scope =>
      (waitForAllThreads scope state, none)
  | load ptr as =>
      let addr := evalPtr ptr state
      (state, some (readMemory state as addr))
  | store ptr val as =>
      let addr  := evalPtr ptr state
      let v     := evalValue val state
      (writeMemory state as addr v, none)
  | atomic op ptr val as =>
      let addr  := evalPtr ptr state
      let v     := evalValue val state
      let old   := readMemory state as addr
      let new   := applyAtomic op old v
      (writeMemory state as addr new, some old)
```

---

## VERIFICATION OBLIGATIONS

| PO  | Statement                                   | Proof Method                          |
|-----|---------------------------------------------|---------------------------------------|
| PO1 | Index space partition is total and disjoint | Structural induction on lattice arith |
| PO2 | Memory address spaces are disjoint          | Allocator invariant                   |
| PO3 | SIMT divergence always reconverges          | Well-founded termination on div stack |
| PO4 | HB is a strict partial order                | Transitivity + irreflexivity          |
| PO5 | Permission sum ≤ 1 at all addresses         | Monotone permission algebra           |
| PO6 | Barrier redistribution preserves total perm | Conservation law on sep-conj          |
| PO7 | All memory accesses are HB-ordered          | Data-race-freedom from perm algebra   |
| PO8 | Launch body terminates                      | Index space finiteness (Axiom 6)      |

---

## INTEGRATION WITH SOV-RTX

The PAX model is the formal substrate for the sovereign GPU runtime in sov-kernel-monster-rtx:

- **Index Space** → scheduler.cmm grid/block constants (seqs=256, heads=32)
- **SIMT/Warp** → flash_attention.ptx warp-level softmax (shfl.sync.bfly)
- **Permission Algebra** → KV block allocator free-list (each block held by exactly one seq)
- **HB / Fences** → WORM checkpoint every 64 tokens (audit log = HB record)
- **Tensor Ops** → gemm.ptx mma.sync.aligned.m16n8k16 + sampler.c top-p nucleus

```
PAX Layer 0 (Index Space)     → scheduler_janet_array[1] = batch_size
PAX Layer 0 (Memory Model)    → kv_allocator free-list invariant
PAX Layer 1 (SIMT)            → flash_attention_paged EXEC mask
PAX Layer 1 (ISA)             → gemm.ptx ldmatrix + mma.sync
PAX Layer 2 (IR)              → sov_cuda_kernels_init module graph
```
