# Verification record

Date: 2026-09-11. Host: Linux x86-64, GCC 13.3.0.

## Executed locally

- Compiled domain/geometry/graph/GA/serialization and the core test executable with C++20.
- Core checks passed: type/range/angle, exact demand, surplus UNUSED, allocation shape/bounds, proper crossing, endpoint/collinear policy, route deduplication, Hall bottleneck infeasibility, augmenting-path repair, certification rejection, serialization round trip, seeded reproducibility, prior validation, and evaluation caps.
- Compiled and ran `generate_dataset`, `evaluate_ga`, `solve_delivery`.
- Generated 30 synthetic instances with generator/teacher seeds 1000–1029, budget 1000, proposal cap 20000, population 48. 29 certified labels; one excluded from supervised training. Splits are fixed before label generation.
- Test split: 3 instances × seeds 42,43,44 = 9 runs, 9 zero-crossing successes.
- OOD split: 3 larger instances × seeds 42,43,44 = 9 runs, 6 zero-crossing successes. Three runs exhausted the 1000-evaluation cap with best crossing count 1. These failures are retained in CSV.
- `solve_delivery` exported the example with feasibility=1, crossing=0, evaluations=3.

CSV files are actual measured output, with wall times specific to this small synthetic environment. They do not show that GNN guidance improves GA.

## Dependency limits and CI

CMake and LibTorch are absent locally. A LibTorch download attempt was blocked by the environment's network approval mechanism. Therefore the LibTorch translation units, neural tests, training and matched GNN-GA experiment have **not been compiled/executed locally**. No trained checkpoint or neural improvement is claimed.

`.github/workflows/cpp.yml` contains a core CMake job and a pinned CPU LibTorch job including neural tests and a complete C++ generate/train/evaluate smoke. Consult Actions for the exact commit's outcome; presence of a workflow is not proof that it passed.

The local sanitizer build uses AddressSanitizer and UndefinedBehaviorSanitizer. LeakSanitizer cannot operate under this environment's tracing layer; leak detection is disabled for that run. This does not establish absence of leaks.

## Reproduce the core build without CMake

```bash
mkdir -p build-core
g++ -std=c++20 -O2 -Wall -Wextra -Werror -Wpedantic -I include src/domain.cpp src/geometry.cpp src/graph.cpp src/ga.cpp src/io.cpp tests/test_core.cpp -o build-core/test_core
build-core/test_core
```

The successful remote verification below closes the initial LibTorch build/test gate. The remaining research task is testing guidance on larger, realistic held-out instances with multiple seeds.

## Successful remote LibTorch verification

[GitHub Actions run 34558340224](https://github.com/dungTudonghoa/delivery/actions/runs/34558340224), commit `c84ab96d194349ae6a4a6a74ba51cee08e84dad4`, completed successfully on Ubuntu 24.04 with GCC 13.3 and CPU LibTorch 2.7.1:

- CMake core-only build/test: passed.
- CMake full build of all targets: passed.
- Core and GNN CTest suites: **2/2 passed**. Neural tests cover variable shapes, receiver/slot permutation equivariance, masked normalization, UNUSED/no-edge handling, learning-loss decrease with encoder gradients, and checkpoint round trip.
- Full C++ smoke generation/training/matched evaluation: passed.
- Validation NLL for the two smoke epochs: **1.02569 → 0.971827** (hidden=16, layers=2, batch=4, seed=42, learning rate=0.001).
- [CSV artifact](https://github.com/dungTudonghoa/delivery/actions/runs/34558340224/artifacts/10183417515) contains the matched evaluation and training log.

This closes the initial LibTorch compilation/runtime gap for that commit. The subsequent module separation is also covered by the same workflow on its own commit. Two epochs on tiny synthetic data are an integration check, not evidence of improved optimization performance.
