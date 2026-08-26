## MultiUUV-Link Showcase

### Communication-Aware Decentralized Multi-UUV Mission

![MultiUUV-Link fault-recovery showcase](artifacts/readme_showcase/multiuuv_fault_recovery_showcase.gif)

The showcase demonstrates four UUVs operating with local perception,
persistent private knowledge, intermittent communication, distributed task
allocation, collision-aware motion, learned frontier ranking, and
failure-aware mission recovery.

### Controller Benchmark

| Controller | Final Coverage | Coverage AUC | Redundancy | Safety Overrides | Mean Minimum Separation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical | 97.74% | 16179.2 | 0.256 | 31.4 | 129.88 px |
| Learned | 96.69% | 16117.8 | 0.275 | 34.4 | 130.64 px |
| Hybrid | 97.49% | 16316.0 | 0.286 | 19.0 | 130.35 px |

### Fault-Recovery Demonstration

| Metric | Result |
| --- | ---: |
| Physical failure time | 70 |
| Majority confirmation time | 232 |
| Detection delay | 162 steps |
| Recovery UUV | UUV-2 |
| Recovery completed | True |
| Completion time | 243 |
| Total recovery delay | 173 steps |
| Failed-UUV displacement | 0.00 px |
| Final coverage | 98.43% |
| Resources discovered | 10/11 |
| Minimum separation | 130.03 px |
| Safety overrides | 29 |
| False-positive failures | 0 |

> Results above are generated from a fresh modular execution and may differ
> numerically from the original notebook demonstration because the historical
> notebook contained stochastic states that were not fully checkpointed.
