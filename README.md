# **MultiUUV-Link**

![Version](https://img.shields.io/badge/version-v0.1.0-blue)
![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![Status](https://img.shields.io/badge/status-validated%20research%20prototype-success)
![License](https://img.shields.io/badge/license-Peacock%20Dynamics%20Proprietary-red)

**Communication-aware decentralized multi-UUV mission autonomy under partial perception, degraded communication, safety constraints, and vehicle failure.**

Developed by **Peacock Dynamics**.

---

## Abstract

**MultiUUV-Link** is an algorithm-first research prototype for decentralized cooperative autonomy among multiple Uncrewed Underwater Vehicles (UUVs). It studies a deliberately narrow but difficult robotics problem: how a team of autonomous underwater agents can explore an initially unknown seabed, discover hidden resources, exchange knowledge when communication is available, distribute exploration work, avoid unsafe proximity, adapt to communication degradation, and continue the mission after a vehicle failure, all without giving the autonomy stack continuous access to simulator ground truth.

The prototype uses a lightweight two-dimensional environment inherited conceptually from the MicroTrench-2D philosophy. Each UUV is represented as a point agent operating over a static seabed image. Every vehicle receives only a bounded local Field of View (FOV), maintains persistent private exploration and resource memory, and acquires remote knowledge only through explicit communication. The central information constraint is therefore

$$
K_i(t) \neq K_{\text{world}}
$$

in general, where $K_i(t)$ is the knowledge available to UUV $i$ at time $t$ and $K_{\text{world}}$ is complete simulator ground truth.

The autonomy stack combines classical frontier exploration, dynamic range-limited communication, one-hop knowledge exchange, distributed frontier-task allocation, adaptive replanning, a predictive multi-UUV safety shield, degraded communication with packet loss, latency and blackouts, heartbeat-based fault detection, orphan-task recovery, and a PyTorch frontier-value model used through a confidence-gated hybrid controller.

The final notebook prototype was developed incrementally, validated experimentally, benchmarked across Classical, Learned, and Hybrid controllers, and then behavior-preservingly modularized into the `multiuuv_link/` Python package. The canonical prototype artifacts report a Hybrid mean final coverage of **98.63%**, mean coverage AUC of **16006.52**, mean redundancy of **0.2337**, and a fault-recovery demonstration in which a physically failed UUV remained immobile, the failure was confirmed after delayed heartbeat evidence, its orphaned exploration task was reassigned, all **10/10** resources were discovered, and minimum separation remained **131.52 px**, above the required **128 px** safety threshold.

This is **not** a high-fidelity marine simulator. It is a controlled autonomy research platform designed to expose the information, coordination, safety, learning, and resilience structure of multi-UUV missions before hydrodynamics and hardware complexity are introduced.

---

## Table of Contents

- [Introduction](#introduction)
- [Core Research Question](#core-research-question)
- [Design Philosophy](#design-philosophy)
- [Mathematical Modeling](#mathematical-modeling)
- [How the System Was Built](#how-the-system-was-built)
- [Final Autonomy Architecture](#final-autonomy-architecture)
- [Repository Structure](#repository-structure)
- [Canonical Prototype Outputs](#canonical-prototype-outputs)
- [What We Achieved](#what-we-achieved)
- [Learned Frontier Model](#learned-frontier-model)
- [Controller Benchmark](#controller-benchmark)
- [Hybrid Authority Analysis](#hybrid-authority-analysis)
- [Fault-Recovery Showcase](#fault-recovery-showcase)
- [How to Reproduce the Results](#how-to-reproduce-the-results)
- [Why This Matters](#why-this-matters)
- [Conclusion](#conclusion)
- [Honest Limitations](#honest-limitations)
- [Probable Extensions](#probable-extensions)
- [License](#license)

---

# Introduction

Autonomous underwater exploration is fundamentally an **information-constrained robotics problem**.

A real UUV does not begin with a complete God's-eye description of its surroundings. Knowledge must be accumulated through sensing, motion, memory, communication, and inference. A multi-UUV mission adds another layer of difficulty because each vehicle may possess a different, incomplete, and potentially stale view of the world.

A single-agent exploration loop can be summarized as:

```text
Observe
  ↓
Move
  ↓
Observe again
```

A decentralized multi-UUV mission is structurally different:

```text
Local Observation
       ↓
Private Memory
       ↓
Local Decision
       ↓
Communication Opportunity
       ↓
Knowledge Exchange
       ↓
Distributed Coordination
       ↓
Safety Constraints
       ↓
Motion
       ↓
Environment Changes Relative to Agent
       ↓
Repeat
```

The challenge is therefore not simply moving four markers over an image. The difficult part is preserving the distinction between:

```text
what exists in the simulated world
```

and:

```text
what each UUV legitimately knows exists
```

MultiUUV-Link is built around that distinction.

The simulator may know all UUV positions, all resources, all failures, and the complete map so that it can render the experiment and evaluate performance. The autonomy stack is not allowed to silently access that complete state. Each agent must build its own world model from local observations and explicit messages.

That constraint makes the project useful as a research sandbox for decentralized robotics rather than merely a multi-agent animation.

---

# Core Research Question

The project asks:

> **How can multiple UUVs cooperatively explore an initially unknown seabed, discover hidden resources, exchange useful knowledge under limited or intermittent communication, distribute tasks without relying on continuous omniscient centralized control, adapt their plans when new information arrives, avoid each other, and continue the mission when communication links or vehicles fail?**

This produces several subordinate research questions:

| Area | Question |
| --- | --- |
| Exploration | Can several UUVs cover an unknown environment effectively without a global planner? |
| Information | Can every vehicle maintain a coherent local world model without simulator-ground-truth leakage? |
| Coordination | Can explicit communication reduce duplicated exploration? |
| Allocation | Can frontier work be distributed using only information available within the connected team? |
| Communication | How does limited range, packet loss, delay, and blackout affect cooperation? |
| Safety | Can exploration intent be separated from collision-prevention execution? |
| Resilience | Can healthy UUVs continue and recover useful work after another vehicle fails? |
| Learning | Can a learned frontier-value model contribute under exactly the same information constraints as the classical controller? |

---

# Design Philosophy

MultiUUV-Link follows four principles.

## 1. Partial information is a hard architectural constraint

The simulator is allowed to be omniscient. The agents are not.

For UUV $i$:

$$
K_i(t) \subseteq K_{\text{world}}
$$

and usually:

$$
K_i(t) \neq K_{\text{world}}.
$$

Agent knowledge evolves only through legitimate sensing and communication.

## 2. Autonomy first, physics later

The project deliberately does not begin with hydrodynamics, 6-DOF rigid-body equations, realistic sonar propagation, current fields, thruster models, or detailed vehicle geometry.

The simulator instead preserves the structural problems needed to study:

- exploration;
- decentralized knowledge;
- communication;
- task distribution;
- replanning;
- safety;
- fault recovery;
- learned decision support.

## 3. Classical methods remain the reference baseline

Learning is not included merely to place machine learning in the repository. A learned component must solve a defined decision problem, obey the same information restrictions as the classical policy, and survive a controlled comparison.

The comparison philosophy is:

$$
\boxed{
\text{Classical Baseline}
\rightarrow
\text{Learned Variant}
\rightarrow
\text{Controlled Comparison}
\rightarrow
\text{Hybrid Decision}
}
$$

## 4. Build, validate, benchmark, then modularize

The original prototype was intentionally developed in Jupyter as an executable specification and visual debugger before stable behavior was extracted into modules.

The overall engineering path was:

$$
\boxed{
\text{Build}
\rightarrow
\text{Validate}
\rightarrow
\text{Benchmark}
\rightarrow
\text{Learn}
\rightarrow
\text{Modularize}
}
$$

The notebook used incremental Markdown-plus-code development units so that every new mechanism was specified, implemented, executed, inspected, and validated before the next layer was introduced.

---

# Mathematical Modeling

## 1. Environment

Let the two-dimensional operating domain be

$$
\Omega
=
\{(x,y)\mid 0\le x<W,\;0\le y<H\},
$$

where $W$ and $H$ are the map width and height.

The canonical seabed image is

```text
assets/sample_map.png
```

with validated dimensions:

$$
W=1448,\qquad H=1086.
$$

The image is represented as

$$
I:\Omega\rightarrow\mathbb{R}^{3}.
$$

The RGB/BGR pixel values provide visual texture for the simulation and visualization. Pixel coordinates are not assumed to correspond to calibrated physical meters.

---

## 2. UUV state

For UUV $i$, the minimum kinematic state is

$$
s_i(t)=(x_i(t),y_i(t)),
$$

or equivalently

$$
p_i(t)=
\begin{bmatrix}
x_i(t)\\
y_i(t)
\end{bmatrix}.
$$

The lightweight prototype deliberately excludes velocity, acceleration, heading, angular rate, depth, roll, and pitch from the vehicle state.

The initial swarm contains

$$
N=4
$$

UUVs.

---

## 3. Discrete action model

The mission planner uses four cardinal actions:

$$
a_i(t)\in\{0,1,2,3\}
$$

with

$$
0=\text{Up},\quad
1=\text{Right},\quad
2=\text{Down},\quad
3=\text{Left}.
$$

For operational motion step

$$
\Delta=32\text{ px},
$$

the displacement vectors are

$$
u_0=\begin{bmatrix}0\\-\Delta\end{bmatrix},\quad
u_1=\begin{bmatrix}\Delta\\0\end{bmatrix},\quad
u_2=\begin{bmatrix}0\\\Delta\end{bmatrix},\quad
u_3=\begin{bmatrix}-\Delta\\0\end{bmatrix}.
$$

The next state is

$$
p_i(t+1)
=
\operatorname{Clamp}
\left(p_i(t)+u_{a_i(t)},\Omega\right).
$$

An additional `HOLD` action exists only inside the predictive safety layer. It is not part of the normal exploration policy.

---

## 4. Local perception

Each UUV observes only a bounded local Field of View:

$$
o_i(t)=\operatorname{Crop}(I,p_i(t),FOV_i).
$$

The validated FOV size is

$$
128\times128\text{ px}.
$$

The observation is padded at map boundaries so that the output remains fixed-size. Padding is not counted as observed world area.

The key information restriction is

$$
o_i(t)\neq I
$$

except in the degenerate case where the FOV covers the full environment.

---

## 5. Hidden resource model

Resource $j$ is represented conceptually as

$$
R_j=(x_j,y_j,c_j,d_j),
$$

where $(x_j,y_j)$ is the hidden ground-truth position, $c_j$ is a generic class, and $d_j$ is discovery state.

The prototype uses generic classes

$$
c_j\in\{A,B,C\}.
$$

These classes are intentionally not geological claims.

Detection is deterministic in the prototype. If a resource lies inside the current FOV,

$$
R_j.position\in FOV_i(t),
$$

then local discovery changes from unknown to known:

$$
d_{ij}^{local}:0\rightarrow1.
$$

Thus

$$
P(\text{detect}\mid R_j\in FOV_i)=1,
$$

and

$$
P(\text{detect}\mid R_j\notin FOV_i)=0.
$$

---

## 6. Agent knowledge and private memory

The simulator possesses full world truth $K_{\text{world}}$, but UUV $i$ may know only:

1. what it currently observes;
2. what it observed previously;
3. what another UUV successfully communicated.

The knowledge update is

$$
K_i(t+1)
=
K_i(t)
\cup O_i(t)
\cup C_i(t),
$$

where $O_i(t)$ is newly observed information and $C_i(t)$ is successfully received communication.

Private exploration memory evolves monotonically:

$$
M_i(t+1)
=
M_i(t)\cup FOV_i(t).
$$

In general,

$$
M_i(t)\neq M_j(t),\qquad i\neq j.
$$

Communication can increase what an agent knows, but it does not retroactively transform another agent's locally observed map into its own local experience. The implementation therefore distinguishes **local exploration memory** from **total known exploration information**.

---

## 7. Frontier-based exploration

A frontier is the boundary between observed and unobserved space in an agent's available planning knowledge.

For candidate frontier goal $g$, the classical utility is

$$
J_i(g)
=
w_I I_i(g)
-
w_D D_i(g)
-
w_R R_i(g),
$$

where:

- $I_i(g)$ is estimated information gain;
- $D_i(g)$ is normalized travel distance;
- $R_i(g)$ is estimated redundancy.

The validated weights are

$$
w_I=1.00,\qquad
w_D=0.35,\qquad
w_R=0.25.
$$

The selected classical goal is

$$
g_i^*=\arg\max_g J_i(g).
$$

Frontier candidates are spatially sampled with a nominal spacing of **128 px**, and the final planner evaluates up to **12** candidates where specified by the validated implementation.

This creates an interpretable baseline: prefer frontier regions with high unknown content, while penalizing travel distance and revisiting already-known space.

---

## 8. Communication graph

Communication is represented as a dynamic graph

$$
G_t=(V,E_t),
$$

where

$$
V=\{1,2,\ldots,N\}.
$$

A communication edge exists when

$$
(i,j)\in E_t
\iff
\|p_i(t)-p_j(t)\|_2\le R_c,
$$

with communication range

$$
R_c=400\text{ px}.
$$

The neighbor set of UUV $i$ is therefore

$$
\mathcal{N}_i(t)=\{j:(i,j)\in E_t\}.
$$

The graph changes with vehicle geometry.

Knowledge exchange is explicitly message-mediated. One communication round uses pre-round snapshots, which prevents information received during a round from being immediately retransmitted in the same round. In other words, communication is one-hop per round rather than an instantaneous connected-component union.

---

## 9. Degraded communication

Connectivity and successful message delivery are intentionally separated.

A geometric edge may exist while an individual packet is lost:

$$
(i,j)\in E_t
\not\Rightarrow
\text{successful packet delivery}.
$$

The validated packet-loss probability is

$$
P_{loss}=0.25,
$$

so nominal packet success probability is

$$
P_{success}=1-P_{loss}=0.75.
$$

Communication latency is represented by

$$
t_{receive}=t_{send}+\tau,
$$

with

$$
\tau\in[1,5]\text{ steps}.
$$

Persistent link blackouts are also modeled. The validated blackout-start probability is

$$
P_{blackout,start}=0.02,
$$

with blackout durations sampled in the range **10 to 30 steps**.

The model is intentionally abstract. It represents the operational consequence of unreliable underwater communication without claiming to simulate acoustic propagation physics.

---

## 10. Distributed frontier-task allocation

Communication-connected UUVs construct a shared task pool from frontier candidates available within their current knowledge. Candidate tasks are deduplicated spatially, scored for participating agents, and assigned while respecting task uniqueness and minimum goal separation.

Important validated coordination parameters include:

| Parameter | Value |
| --- | ---: |
| Stable allocation interval | 12 steps |
| Allocation cooldown | 4 steps |
| Minimum useful unknown ratio | 0.15 |
| Minimum assigned-goal separation | 128 px |
| Task-overlap weight | 0.65 |
| Goal-reached radius | 32 px |

The objective is not to emulate a full market or globally synchronized auction. The prototype instead validates a lightweight distributed allocation mechanism compatible with partial knowledge and changing connectivity.

---

## 11. Predictive safety shield

Mission planning and safety are separated.

For UUVs $i$ and $j$,

$$
d_{ij}(t)=\|p_i(t)-p_j(t)\|_2.
$$

The required operational separation is

$$
d_{safe}=128\text{ px}.
$$

The planner proposes desired actions $a_i^{des}$, while the safety layer may execute different actions $a_i^{exec}$ if the desired joint motion is unsafe.

For a joint action vector

$$
\mathbf{a}=(a_0,a_1,\ldots,a_{N-1}),
$$

the predicted next position is

$$
p_i^+=f(p_i,a_i).
$$

A joint action is feasible only if

$$
\|p_i^+-p_j^+\|_2\ge d_{safe}
$$

for every pair $i\neq j$.

The safety candidate set is

$$
A_{safety}=\{UP,RIGHT,DOWN,LEFT,HOLD\}.
$$

For four UUVs this produces

$$
5^4=625
$$

possible joint safety actions, small enough for exhaustive evaluation.

The validated shield uses a lexicographic objective:

1. minimize the number of overrides relative to planner intent;
2. among equally invasive safe solutions, maximize predicted minimum pairwise separation.

This architecture keeps exploration intent distinct from safety-constrained execution.

---

## 12. Fault detection and recovery

A failed UUV transitions from active mission participation to a physically stationary obstacle.

Heartbeat-based failure suspicion is local and contact-aware. For detector $i$ and peer $f$, suspicion increases only when communication geometry indicates contact should be possible but a fresh heartbeat is absent.

The validated local threshold is

$$
S_{if}\ge40.
$$

Swarm-level failure confirmation requires

$$
N_{confirm}=2
$$

independent detections.

The final fault showcase injects physical failure of **UUV-1** at

$$
t_f=70.
$$

After physical failure, the failed UUV:

- stops moving;
- stops sensing;
- stops transmitting;
- stops receiving;
- remains a stationary object for safety reasoning.

After majority confirmation, the system attempts to preserve mission intent by creating or recovering an orphaned frontier task, selecting a healthy successor, temporarily prioritizing that task, and declaring task completion when

$$
d\le32\text{ px}.
$$

A simple resilience abstraction is

$$
R=
\frac{Performance_{failure}}
{Performance_{nominal}},
$$

with the desired qualitative behavior being graceful degradation rather than mission collapse.

---

## 13. Learned frontier-value model

The learned component is a supervised PyTorch frontier-value estimator. It is trained to approximate the classical frontier utility using only features available to the decentralized planner.

The feature vector is

$$
x(g)=
\begin{bmatrix}
\text{information ratio}\\
\text{normalized distance}\\
\text{redundancy ratio}\\
\text{local coverage}\\
\text{known coverage}\\
\text{normalized communication degree}
\end{bmatrix}.
$$

The target is

$$
y(g)=J_{classical}(g).
$$

The network architecture is

$$
6\rightarrow32\rightarrow32\rightarrow1
$$

with ReLU activations after both hidden layers.

Validated training configuration:

| Item | Value |
| --- | --- |
| Dataset seed | `370037` |
| Synthetic states per UUV | 80 |
| Maximum candidates per dataset state | 20 |
| Retention probability range | 0.20 to 1.00 |
| Split | 70% train / 15% validation / 15% test |
| Grouping | `(uuv_id, synthetic_state)` |
| Normalization | train split only |
| Batch size | 256 |
| Epochs | 150 |
| Optimizer | Adam |
| Learning rate | $10^{-3}$ |
| Loss | MSE |
| Train seed | `380038` |

Because the model learns the classical scoring target, high regression accuracy demonstrates **faithful approximation of that target**, not proof that the learned policy is inherently superior to the classical planner.

---

## 14. Confidence-gated Hybrid controller

The final controller does not blindly replace the classical planner with the neural network.

For the same ordered candidate set, the Classical and Learned rankers select their preferred goals. If they agree, the choice is recorded as a learned agreement.

If they disagree, define the learned confidence margin as the difference between the highest and second-highest learned utility values:

$$
m_{learned}
=
\hat J_{(1)}-\hat J_{(2)}.
$$

Define classical regret for selecting the learned-preferred candidate as

$$
r_{classical}
=
J_{classical}^{best}
-
J_{classical}(g_{learned}).
$$

A learned override is authorized only if

$$
m_{learned}\ge0.03
$$

and

$$
r_{classical}\le0.02.
$$

Otherwise the controller falls back to the classical choice.

Conceptually:

$$
\boxed{
\text{Classical Autonomy}
+
\text{PyTorch Learned Advisor}
+
\text{Confidence Gate}
}
$$

This keeps the learned model inside a bounded decision architecture rather than giving it unconstrained authority.

---

# How the System Was Built

## Jupyter-first development

The complete research prototype was first built in:

```text
notebooks/MultiUUV_Link_Prototype.ipynb
```

The accompanying author implementation record is:

```text
notebooks/MultiUUV_Link_Prototype_Author_Implementation.pdf
```

The notebook acted simultaneously as:

- executable specification;
- mathematical notebook;
- algorithm-development environment;
- visual debugger;
- experiment workspace;
- architecture prototype;
- record of engineering decisions.

Development proceeded incrementally. Each new behavior was introduced only after the previous behavior had been run and inspected.

The effective progression was:

```text
Static seabed environment
        ↓
Point-UUV state and cardinal motion
        ↓
Fixed-size local FOV perception
        ↓
Hidden resources and deterministic discovery
        ↓
Persistent private exploration memory
        ↓
Four independent UUVs
        ↓
Frontier extraction and classical scoring
        ↓
Closed-loop autonomous exploration
        ↓
Predictive multi-UUV safety shield
        ↓
Range-limited dynamic communication graph
        ↓
Explicit one-hop knowledge exchange
        ↓
Distributed frontier-task allocation
        ↓
Adaptive replanning
        ↓
Packet loss, latency, and blackouts
        ↓
Vehicle failure injection
        ↓
Heartbeat-based distributed fault confirmation
        ↓
Orphan-task recovery and mission continuation
        ↓
Classically supervised PyTorch frontier model
        ↓
Learned controller
        ↓
Confidence-gated Hybrid controller
        ↓
Paired controller benchmark
        ↓
Normal-mission and fault-recovery GIFs
        ↓
Metrics, JSON summary, and model export
        ↓
Behavior-preserving Python modularization
```

---

## Step 1: Rebuild the lightweight seabed world

The project reconstructs the required MicroTrench-style environment directly around `assets/sample_map.png`. The environment stores the static seabed image, world dimensions, hidden resource positions, and true UUV positions.

This global state is used for simulation, rendering, and evaluation. It is not directly exposed to the autonomy logic.

---

## Step 2: Validate local perception before cooperation

The first autonomy milestone is deliberately modest: each UUV must receive only its own `128 x 128` FOV and maintain its own history.

This stage validates the most important boundary in the project before communication is introduced.

If every agent were allowed to read the global map, later communication and coordination experiments would become scientifically meaningless.

---

## Step 3: Build persistent private world models

Each UUV accumulates its own exploration mask and resource memory. The memory is monotonic under normal mission execution: once legitimately observed, a cell remains known to that UUV.

The system separately tracks knowledge acquired locally and knowledge acquired remotely. This makes it possible to evaluate what communication actually contributes.

---

## Step 4: Establish an interpretable classical exploration baseline

Frontier extraction identifies unknown cells adjacent to explored space. Spatial sampling converts potentially thousands of frontier pixels into a manageable candidate set. Each candidate is scored using information gain, travel distance, and redundancy.

This creates a transparent baseline whose behavior can be inspected numerically and visually.

---

## Step 5: Insert safety between planning and execution

The frontier planner expresses where each vehicle wants to move. The predictive safety shield decides whether those motions may be executed simultaneously.

For the four-agent prototype, exhaustive evaluation of 625 joint safety actions is computationally tractable and avoids hidden pairwise heuristics.

This stage establishes the invariant:

$$
d_{min}^{mission}\ge128\text{ px}.
$$

---

## Step 6: Add communication without collapsing decentralization

Communication begins as geometry-dependent graph connectivity. Agents exchange snapshots of what they actually know.

The one-hop, pre-round snapshot rule is important: information cannot teleport through an entire connected component within one message round.

A disconnected UUV therefore remains genuinely disconnected until it reconnects or obtains information through a later valid exchange.

---

## Step 7: Coordinate frontier work

Connected vehicles construct and score frontier task pools, remove spatially redundant tasks, and assign goals while maintaining goal separation.

The allocation layer is periodically updated rather than continuously replacing goals, which reduces unstable goal thrashing while still permitting replanning when information or connectivity changes.

---

## Step 8: Degrade the communication channel

The project then separates the existence of a communication link from the success of a specific message.

Packet loss, delay, and persistent blackouts are introduced while preserving local autonomy. A UUV should continue useful exploration even if cooperation temporarily degrades.

---

## Step 9: Introduce physical vehicle failure

A permanent vehicle failure is injected into an otherwise functioning mission. The failed UUV stops participating in sensing, planning, and communication.

Healthy agents do not receive an omniscient global `vehicle_failed=True` signal. Instead, failure is inferred through missing heartbeat evidence under communication-contact conditions and confirmed through multiple independent detections.

---

## Step 10: Recover abandoned mission work

After failure confirmation, the system creates or recovers an orphaned frontier task and selects a healthy successor. The successor temporarily prioritizes the orphaned goal until completion.

This converts failure handling from a visual event into a mission-level recovery behavior.

---

## Step 11: Learn the frontier-value function

A supervised dataset is generated from decentralized frontier candidates. The six-feature PyTorch network learns the classical frontier utility.

The model is evaluated not only with regression metrics but also with ranking-oriented metrics because a planner ultimately cares about the ordering of candidate goals.

---

## Step 12: Compare Classical, Learned, and Hybrid controllers

All three controllers are evaluated under matched stochastic conditions across five paired seeds and 180 mission steps per run.

The experiment records:

- final coverage;
- coverage AUC;
- redundancy;
- hold actions;
- safety overrides;
- goal changes;
- minimum separation.

The final design retains the confidence-gated Hybrid architecture because the canonical benchmark achieved the strongest overall mission-level combination of final coverage, coverage AUC, redundancy, and coverage stability.

---

## Step 13: Export evidence, then modularize

Only after the notebook behavior was validated were stable components extracted into narrowly responsible Python modules.

The modular package preserves the central invariants:

- no simulator-ground-truth leakage into autonomy;
- `(x, y)` geometry and `[y, x]` array indexing;
- fixed-size FOVs with valid-world masking;
- private persistent local memories;
- one-hop communication semantics;
- `128 px` operational safety separation;
- failed-UUV immobility;
- common candidate ordering for Classical and Learned rankers;
- confidence-gated Hybrid regret bounds;
- non-mutating visualization and reporting.

---

# Final Autonomy Architecture

The validated mission stack can be summarized as:

```text
              SIMULATOR / GROUND TRUTH
        seabed, hidden resources, true positions
                       │
                       │ local observation only
                       ▼
              LOCAL FOV PERCEPTION
                       │
                       ▼
          PERSISTENT PRIVATE MEMORY
                       │
                       ▼
               FRONTIER EXTRACTION
                       │
                       ▼
       COMMUNICATION-AWARE KNOWLEDGE
                       │
                       ▼
        DISTRIBUTED FRONTIER ALLOCATION
                       │
                       ▼
       CLASSICAL + LEARNED RANKING
              CONFIDENCE GATE
                       │
                       ▼
          PREDICTIVE SAFETY SHIELD
                       │
                       ▼
             DISCRETE UUV MOTION
                       │
                       └──────────────┐
                                      │
                     observe, communicate, replan
                                      │
                                      └──────────────► repeat
```

Communication degradation sits across the communication path:

```text
Geometric range
     ↓
Persistent blackout
     ↓
Packet loss
     ↓
Latency
```

Fault recovery adds:

```text
Physical UUV failure
        ↓
Heartbeat silence
        ↓
Contact-aware suspicion
        ↓
Majority confirmation
        ↓
Orphaned frontier task
        ↓
Healthy successor selection
        ↓
Task reassignment
        ↓
Mission continuation
```

---

# Repository Structure

```text
MultiUUV-Link/
│
├── artifacts/                                  # Canonical prototype outputs
│   ├── frontier_value_model_prototype.pt       # Exported learned frontier-value model
│   ├── multiuuv_fault_recovery_showcase_prototype.gif
│   │                                           # 260-step decentralized fault-recovery showcase
│   ├── multiuuv_final_metrics_prototype.csv    # Final Classical/Learned/Hybrid benchmark table
│   ├── multiuuv_final_summary_prototype.json   # Architecture, learning, benchmark, hybrid, and fault metrics
│   ├── multiuuv_hybrid_mission_prototype.gif   # 180-step normal Hybrid cooperative mission
│   └── readme_showcase/                        # Supplemental fresh modular rerun outputs
│       ├── controller_benchmark_runs.csv
│       ├── controller_benchmark_summary.csv
│       ├── fault_recovery_metrics.json
│       ├── multiuuv_fault_recovery_showcase.gif
│       └── README_SHOWCASE.md
│
├── assets/
│   └── sample_map.png                          # Static 1448 x 1086 seabed environment
│
├── experiments/
│   ├── generate_showcase_artifacts.py          # Generates a fresh modular README showcase bundle
│   ├── output/
│   │   └── frontier_value_model.pt             # Fresh model produced by training script
│   ├── run_benchmarks.py                       # Paired Classical/Learned/Hybrid benchmark runner
│   ├── run_fault_recovery.py                   # Modular fault-recovery mission runner
│   └── train_frontier_model.py                 # Reproduces classically supervised model training
│
├── multiuuv_link/
│   ├── __init__.py                             # Package interface
│   ├── config.py                               # Numerical configuration, paths, seeds, thresholds
│   ├── environment.py                          # Seabed loading, dimensions, resource generation
│   ├── models.py                               # UUVState, Resource, UUVAgent data structures
│   ├── motion.py                               # Cardinal motion, clamping, trajectory geometry
│   ├── perception.py                           # FOV extraction and visible-resource discovery
│   ├── memory.py                               # Private exploration/resource/planning memories
│   ├── frontier.py                             # Frontier masks, candidates, classical scoring
│   ├── safety.py                               # Predictive joint-action safety shield
│   ├── communication.py                        # Range graph, messaging, loss, delay, blackout
│   ├── allocation.py                           # Task pools, deduplication, assignment, replanning
│   ├── fault_recovery.py                       # Heartbeats, suspicion, confirmation, recovery task logic
│   ├── learning.py                             # Dataset, MLP training, metrics, model serialization
│   ├── controllers.py                          # Classical, Learned, and confidence-gated Hybrid policies
│   ├── missions.py                             # Closed-loop mission orchestration
│   ├── benchmarks.py                           # Paired experiments and aggregate metrics
│   ├── visualization.py                        # Static plots, mission frames, GIF generation
│   ├── artifacts.py                            # CSV/JSON/model export and artifact validation
│   └── validation.py                           # Reusable invariant and end-to-end checks
│
├── notebooks/
│   ├── MultiUUV_Link_Prototype.ipynb           # Authoritative executable prototype
│   └── MultiUUV_Link_Prototype_Author_Implementation.pdf
│                                               # Full theory + executed implementation record
│
├── tests/
│   ├── conftest.py
│   ├── test_benchmarks_fault_artifacts.py      # Benchmark, fault, export regression coverage
│   ├── test_communication_allocation.py        # Communication and distributed allocation checks
│   ├── test_environment_motion_perception.py   # Core environment and perception checks
│   ├── test_learning_controllers.py            # Learned model and controller-gate checks
│   ├── test_memory_frontier_safety.py          # Memory, frontier, and safety invariants
│   └── test_source_integrity.py                # Authoritative source-integrity checks
│
├── pyproject.toml                              # Package metadata and Python dependencies
├── requirements.txt                            # Environment dependency list
├── LICENSE                                     # Peacock Dynamics Proprietary License
└── README.md                                   # Project documentation
```

### Source of truth

The authoritative behavioral references are:

1. `notebooks/MultiUUV_Link_Prototype.ipynb`
2. `notebooks/MultiUUV_Link_Prototype_Author_Implementation.pdf`

The modular Python package is a behavior-preserving extraction of the validated notebook architecture, not a redesign of it.

---

# Canonical Prototype Outputs

The primary README results below come from these files directly under `artifacts/`:

```text
artifacts/multiuuv_hybrid_mission_prototype.gif
artifacts/multiuuv_fault_recovery_showcase_prototype.gif
artifacts/multiuuv_final_metrics_prototype.csv
artifacts/multiuuv_final_summary_prototype.json
artifacts/frontier_value_model_prototype.pt
```

The `artifacts/readme_showcase/` directory contains a later fresh modular rerun. Because some early notebook random state was not fully checkpointed, fresh modular executions can differ numerically from the canonical notebook artifacts. The headline values in this README intentionally use the canonical `multiuuv_final_metrics_prototype.csv` and `multiuuv_final_summary_prototype.json` results.

---

# What We Achieved

## 1. Normal Hybrid cooperative mission

![MultiUUV-Link Hybrid Cooperative Underwater Exploration](artifacts/multiuuv_hybrid_mission_prototype.gif)

The normal mission visualization demonstrates the complete Hybrid autonomy loop without physical vehicle failure.

The global visualization may display information useful for debugging, including UUV positions, FOV rectangles, frontier goals, communication links, trajectories, discovered resources, and mission metrics. That display is **not** the knowledge available to any individual vehicle.

The architectural rule remains:

$$
\boxed{
\text{Visualization State}
\neq
\text{Agent Knowledge}
}
$$

The Hybrid controller combines the classical frontier score with learned frontier-value estimates through the confidence gate, then passes the desired joint motion through the predictive safety shield before execution.

---

## 2. Decentralized fault-recovery mission

![MultiUUV-Link Decentralized Fault-Recovery Mission](artifacts/multiuuv_fault_recovery_showcase_prototype.gif)

The fault-recovery showcase demonstrates the most distinctive system behavior:

$$
\boxed{
\text{Explore}
+
\text{Communicate}
+
\text{Discover}
+
\text{Lose a Vehicle}
+
\text{Detect Failure}
+
\text{Recover Task}
+
\text{Continue Mission}
}
$$

The scenario runs for **260 steps**. UUV-1 physically fails at step **70**. The vehicle then remains immobile and stops participating in normal mission operations. Healthy vehicles continue operating while distributed heartbeat evidence accumulates. The failure is eventually confirmed, an orphaned frontier task is generated from safe mission knowledge, a healthy successor is selected, and the task is completed while preserving the pairwise safety invariant.

---

# Learned Frontier Model

The exported PyTorch model contains **1,313 trainable parameters** and approximates the classical frontier-value function using six decentralized features.

Canonical test-set metrics are:

| Metric | Value |
| --- | ---: |
| Test MSE | $5.9739\times10^{-6}$ |
| Test MAE | 0.001815 |
| Test $R^2$ | **0.999934** |
| Test Pearson correlation | **0.999968** |
| Test Spearman correlation | **0.999955** |
| Top-1 frontier agreement | **95.83%** |
| Model parameters | **1,313** |

These numbers show that the compact MLP learned the intended classical frontier-value mapping extremely closely.

They should be interpreted carefully. The training target is the classical utility itself. Therefore the strong $R^2$, rank correlation, and top-1 agreement demonstrate **model fidelity to the supervised scoring function**. They do not by themselves establish superior mission autonomy.

That question must be answered by closed-loop controller benchmarking.

---

# Controller Benchmark

The final benchmark compares **Classical**, **Learned**, and **Hybrid** controllers over five paired stochastic seeds, with **180 mission steps per controller per seed**, for a total of 15 matched missions.

The canonical metrics are stored in:

```text
artifacts/multiuuv_final_metrics_prototype.csv
```

## Main controller comparison

| Controller | Final Coverage Mean | Final Coverage Std | Coverage AUC | Redundancy Mean | Redundancy Std | Mean HOLDs | Mean Safety Overrides | Mean Goal Changes | Mean Min. Separation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical | 97.4193% | 1.4792% | 15888.997 | 0.2464 | 0.0423 | 55.4 | 54.2 | 48.8 | **134.71 px** |
| Learned | 96.8228% | 2.1480% | 15678.248 | 0.2868 | 0.0411 | **42.6** | **28.4** | 117.2 | 130.37 px |
| **Hybrid** | **98.6256%** | **0.5212%** | **16006.523** | **0.2337** | **0.0172** | 56.0 | 32.2 | 119.8 | 130.44 px |

## Metric-by-metric interpretation

### Final coverage

The Hybrid controller achieved the highest mean final coverage:

$$
Coverage_{Hybrid}=98.6256\%.
$$

Compared with Classical:

$$
\Delta Coverage
\approx
+1.2063\text{ percentage points}.
$$

### Coverage AUC

Hybrid also achieved the highest mean coverage area-under-curve:

$$
AUC_{Hybrid}=16006.523.
$$

This indicates that its advantage is not limited to the final timestep. Across the mission horizon, it accumulated explored area slightly faster overall than the other two controller classes.

### Redundancy

Hybrid achieved the lowest mean exploration redundancy:

$$
\rho_{Hybrid}=0.2337.
$$

Relative to Classical, this is approximately a **5.16% reduction** in mean redundancy.

### Coverage stability

Hybrid produced the smallest final-coverage standard deviation:

$$
\sigma_{Hybrid}=0.5212\%.
$$

This is about **64.8% lower** than the Classical final-coverage standard deviation in the canonical benchmark.

### Safety interventions

The fully Learned controller required the fewest mean safety overrides, at **28.4**. Hybrid required **32.2**, still substantially fewer than Classical at **54.2**.

This matters because the safety shield is not the planner. A high override count indicates that planner intent more often conflicts with the safety layer.

### Minimum separation

All three controller aggregates remained above the required **128 px** safety threshold. Classical had the largest mean minimum separation, while Learned and Hybrid operated closer to the constraint boundary without violating it in the validated benchmark.

---

# Hybrid Authority Analysis

The canonical Hybrid benchmark recorded:

| Decision type | Count |
| --- | ---: |
| Learned agreements | 792 |
| Learned overrides | 4 |
| Classical fallbacks | 60 |
| Total decisions | 856 |
| Learned-authorized fraction | **92.99%** |

This table deserves careful interpretation.

The learned model was authorized on approximately **92.99%** of Hybrid decisions, but **792 of those decisions were agreements with the classical controller**. Only **4** decisions were true learned overrides where the model preferred a different candidate and passed both the confidence-margin and classical-regret gates.

Therefore the Hybrid result should not be described as a neural policy replacing classical planning.

A more accurate description is:

> **The learned model acts as a high-fidelity frontier-value advisor inside a classical safety envelope, with the confidence gate permitting deviation only when learned confidence is sufficient and classical regret remains bounded.**

This is also why the project retains the Classical baseline as a first-class component rather than treating it as disposable once a neural network exists.

---

# Fault-Recovery Showcase

Canonical fault-recovery metrics are stored inside:

```text
artifacts/multiuuv_final_summary_prototype.json
```

The validated result is:

| Metric | Result |
| --- | ---: |
| Mission duration | 260 steps |
| Failed UUV | UUV-1 |
| Physical failure time | 70 |
| Majority confirmation time | 169 |
| Detection delay | **99 steps** |
| Orphan source | `generated_safe_frontier` |
| Orphan goal | `(421, 261)` |
| Recovery UUV | UUV-0 |
| Recovery completion time | 192 |
| Total recovery delay | **122 steps** |
| Failed-UUV displacement after failure | **0.0 px** |
| Final coverage | **96.8154%** |
| Resources discovered | **10 / 10** |
| Safety overrides | 25 |
| Mission minimum separation | **131.52 px** |

## What the fault result validates

### Failure is not globally announced

The physical failure occurs at step 70, but majority confirmation occurs at step 169.

The **99-step detection delay** is not necessarily a weakness of the implementation. It is a consequence of the intentionally constrained information model: healthy agents need sufficient contact-aware heartbeat evidence before the swarm can confirm a peer failure.

### The failed vehicle truly stops

Measured post-failure displacement is

$$
\Delta p_f=0.0\text{ px}.
$$

This confirms that the failed agent does not continue moving through stale planner commands.

### Mission work is recovered

The orphaned frontier target is completed by a healthy successor at step 192.

The system therefore demonstrates mission-level recovery rather than merely marking an agent as failed.

### Healthy agents continue operating

The mission does not terminate when UUV-1 fails. The remaining UUVs continue exploring, communicating when possible, allocating work, and enforcing safety.

### Safety survives the fault

The minimum measured inter-UUV separation is

$$
d_{min}=131.5181\text{ px},
$$

which satisfies

$$
d_{min}>d_{safe}=128\text{ px}.
$$

### Resource discovery remains complete

The showcase ends with

$$
D(T)=\frac{10}{10}=1.0.
$$

All hidden resources in that validated scenario were discovered despite the permanent loss of one vehicle.

---

# Evaluation Metrics

The project uses metrics designed to separate different aspects of mission behavior.

## Coverage

Let $\Omega_{observed}(t)$ be the union of legitimately observed map cells. Then

$$
Coverage(t)
=
\frac{|\Omega_{observed}(t)|}{|\Omega|}.
$$

A time-to-coverage milestone can be expressed as

$$
T_{90}
=
\min\{t:Coverage(t)\ge0.90\}.
$$

## Resource discovery

For $N_{discovered}$ discovered resources out of $N_{total}$:

$$
D(t)=\frac{N_{discovered}(t)}{N_{total}}.
$$

## Exploration redundancy

If $A_i$ is area observed by UUV $i$ and $A_{\cup}$ is the union of all observed regions,

$$
\rho
=
\frac{\sum_i A_i-A_{\cup}}{A_{\cup}}.
$$

Large $\rho$ indicates more repeated exploration.

## Path length

For UUV $i$:

$$
L_i
=
\sum_{t=0}^{T-1}
\|p_i(t+1)-p_i(t)\|_2.
$$

Total abstract swarm path length is

$$
L_{total}=\sum_{i=1}^{N}L_i.
$$

This is a motion-effort proxy only. It is not a calibrated energy model.

## Safety

Mission minimum separation is

$$
d_{min}
=
\min_t\min_{i\neq j}
\|p_i(t)-p_j(t)\|_2.
$$

Additional safety indicators include desired-action violations, safety intervention timesteps, HOLD actions, and total overridden actions.

## Communication

The framework can record packet attempts, deliveries, drops, delivery ratio, latency, graph connectivity, and blackout behavior. These are algorithmic channel metrics, not physical acoustic-link predictions.

---

# How to Reproduce the Results

## Requirements

The package requires **Python 3.10 or newer**.

Core dependencies include:

- NumPy;
- OpenCV;
- Matplotlib;
- NetworkX;
- pandas;
- PyTorch;
- ImageIO;
- Pillow.

Pytest is included in the optional test dependency group.

---

## 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade packaging tools if needed:

```bash
python -m pip install --upgrade pip
```

---

## 2. Install MultiUUV-Link

From the repository root:

```bash
python -m pip install -e ".[test]"
```

---

## 3. Run the regression suite

```bash
pytest -q
```

The tests cover environment behavior, motion, FOV extraction, private memory, frontier logic, predictive safety, communication semantics, allocation, learning, controller gating, fault recovery, artifacts, and source integrity.

---

## 4. Reproduce a fresh learned model from the modular package

```bash
python experiments/train_frontier_model.py \
  --output experiments/output/frontier_value_model.pt
```

This runs the classically supervised frontier-value training procedure and exports a fresh PyTorch checkpoint.

---

## 5. Run a fresh Classical/Learned/Hybrid benchmark

```bash
python experiments/run_benchmarks.py \
  --model experiments/output/frontier_value_model.pt
```

The default benchmark uses 180 mission steps.

Fresh modular runs are expected to preserve algorithms and invariants, but they are not guaranteed to reproduce the historical notebook aggregate values bit-for-bit because some early notebook stochastic state was not fully checkpointed.

---

## 6. Run a fresh fault-recovery mission

```bash
python experiments/run_fault_recovery.py \
  --model experiments/output/frontier_value_model.pt
```

This reports the failure, confirmation, recovery, coverage, discovery, safety, and allocation metrics for a fresh modular execution.

---

## 7. Generate a fresh README-oriented showcase bundle

```bash
python experiments/generate_showcase_artifacts.py
```

This writes a fresh modular showcase under:

```text
artifacts/readme_showcase/
```

including benchmark CSV files, fault-recovery metrics, and a showcase animation.

These supplemental rerun artifacts are intentionally separate from the canonical notebook artifacts used for the headline numbers in this README.

---

## 8. Reproduce the canonical notebook artifact pipeline

The canonical artifacts shown in this README were exported by the validated notebook pipeline itself:

```text
notebooks/MultiUUV_Link_Prototype.ipynb
```

To reproduce that pipeline, open the notebook from the repository root and execute the cells **in order from top to bottom**. The final visualization and artifact-export stages write:

```text
artifacts/multiuuv_hybrid_mission_prototype.gif
artifacts/multiuuv_fault_recovery_showcase_prototype.gif
artifacts/multiuuv_final_metrics_prototype.csv
artifacts/multiuuv_final_summary_prototype.json
artifacts/frontier_value_model_prototype.pt
```

If Jupyter is not installed in the environment:

```bash
python -m pip install jupyter
jupyter notebook notebooks/MultiUUV_Link_Prototype.ipynb
```

### Reproducibility note

The authoritative notebook contains some early stochastic states that were not exported as a complete deterministic checkpoint. Therefore:

- the **algorithms, schemas, validation rules, and architectural invariants** are reproducible;
- the stored canonical artifacts document the validated author run;
- a fresh rerun may produce numerically different resource scenarios, learned checkpoints, controller rankings, or fault timings;
- numerical drift should not be hidden by retuning constants merely to reproduce a historical winner ordering.

For deterministic research releases, future versions should checkpoint all scenario, spawn, motion, communication, training, and fault-recovery random states required to regenerate every reported artifact exactly.

---

# Why This Matters

MultiUUV-Link is intentionally lightweight, but the research structure it preserves maps to several difficult real autonomous-systems questions.

## 1. Decentralized autonomy without omniscient state

Many simulated swarm demonstrations quietly use a global state estimator, shared map, or centralized task planner even when the resulting trajectories look decentralized.

MultiUUV-Link makes the information boundary explicit. That makes it a useful environment for testing whether a coordination mechanism genuinely works under partial knowledge.

## 2. Communication is treated as a resource, not a magical shared memory

The project distinguishes:

- being geometrically connected;
- successfully delivering a message;
- waiting through communication delay;
- losing a link temporarily;
- learning information only after explicit transmission.

This matters for underwater systems because communication can be sparse, slow, intermittent, and expensive.

## 3. Safety remains independent of mission intent

The predictive safety shield allows exploration algorithms to be changed without rewriting collision prevention logic.

That separation is valuable when experimenting with learned or adaptive planners because an exploratory policy can be evaluated behind a consistent safety layer.

## 4. Failure recovery is mission-aware

The project does more than remove a failed agent. It preserves the idea that the vehicle may have owned useful unfinished work and that the remaining swarm should recover that work when possible.

This is closer to real mission resilience than simply decreasing the active-agent count.

## 5. Learning is forced to justify itself

The learned model is evaluated against an interpretable classical baseline under the same information constraints.

The result is nuanced: the model learns the classical frontier-value function extremely well, but the fully Learned controller does not universally dominate Classical in closed-loop behavior. The final Hybrid design therefore uses learning as a bounded advisor rather than assuming that neural control is automatically superior.

That is a more useful engineering conclusion than a benchmark designed only to make the ML method win.

## 6. The framework is suitable for progressive fidelity

Because high-fidelity marine physics are intentionally deferred, algorithm changes can be tested quickly. Once the autonomy architecture is stable, increasingly realistic sensing, communication, dynamics, and hardware interfaces can be introduced without changing the research question all at once.

---

# Conclusion

MultiUUV-Link demonstrates a complete lightweight pipeline for **communication-aware decentralized multi-UUV mission autonomy**.

Within the validated two-dimensional abstraction, four UUV agents:

- operate with bounded local FOV perception;
- maintain persistent private knowledge;
- explore autonomously using frontier-based planning;
- exchange only explicitly transmitted information;
- coordinate through a dynamic communication graph;
- distribute frontier work;
- replan as knowledge and connectivity change;
- preserve minimum separation through a predictive joint-action safety shield;
- continue useful autonomy under packet loss, latency, and communication blackouts;
- detect a physically failed peer through delayed heartbeat evidence;
- create and reassign orphaned mission work;
- use a PyTorch frontier-value model under the same information restrictions as the classical planner;
- compare Classical, Learned, and Hybrid control under matched stochastic conditions;
- export quantitative metrics and visual mission artifacts;
- and retain the final validated behavior in a modular Python package.

The canonical benchmark selected the confidence-gated Hybrid architecture because it achieved the strongest overall combination of final coverage, coverage AUC, redundancy, and coverage stability. The canonical fault showcase further demonstrated safe mission continuation after permanent vehicle loss, including delayed distributed confirmation, successful task reassignment, zero post-failure displacement of the failed UUV, complete resource discovery, and minimum separation above the required threshold.

The main contribution of the project is therefore not a claim of realistic underwater vehicle performance. It is a **technically transparent experimental architecture** for studying how exploration, information, communication, learning, safety, and resilience interact when autonomous agents do not share omniscient knowledge.

---

# Honest Limitations

The limitations below are deliberate and should remain visible whenever the project is presented.

## 1. The world is two-dimensional

All motion occurs in image-space $(x,y)$ coordinates. There is no depth, 3D terrain following, ascent/descent, roll, pitch, yaw, or vertical deconfliction.

## 2. Pixels are not calibrated physical distance

A separation of `128 px`, a communication range of `400 px`, and a motion step of `32 px` are simulator quantities. They are not validated meter-scale specifications.

## 3. UUV dynamics are not modeled

Vehicles are point agents with discrete kinematic moves. The prototype does not model:

- mass or inertia;
- thrusters;
- actuator lag;
- acceleration limits;
- hydrodynamic drag;
- buoyancy;
- currents;
- energy consumption.

Therefore path length and action count are not physical energy predictions.

## 4. Perception is idealized

The FOV is a fixed rectangular crop. Resource detection is deterministic when a resource enters the FOV.

There are no false positives, false negatives, range-dependent confidence, sonar equations, lighting models, turbidity, occlusion, sensor noise, or learned perception.

## 5. Localization is effectively perfect inside the abstraction

Agents act using image-space positions without drift, SLAM uncertainty, dead-reckoning error, map-registration error, or loop-closure failure.

## 6. The seabed image is not a traversability map

The image provides spatial texture and visualization. The current environment does not model realistic terrain collision, cliffs, obstacles, forbidden zones, or bathymetric constraints.

## 7. Resource classes are synthetic

Classes `A`, `B`, and `C` are generic task labels. They do not represent validated geological mineral classes or deposit probabilities.

## 8. Communication is operationally abstract

The communication graph captures range, packet loss, latency, and blackouts, but not real underwater acoustic propagation.

The project does not model:

- sound-speed profiles;
- multipath;
- Doppler;
- attenuation versus frequency;
- modem bandwidth;
- acoustic interference;
- channel coding;
- realistic packet size versus transmission duration;
- energy cost of communication.

## 9. Distributed allocation remains lightweight

The current allocation mechanism is sufficient to study frontier sharing and reassignment, but it is not a complete asynchronous market-based or consensus protocol with formal convergence guarantees under arbitrary communication delay and partition.

## 10. Exhaustive safety search does not scale gracefully

For four UUVs, the safety layer evaluates

$$
5^4=625
$$

joint actions.

For general $N$, the search grows as

$$
5^N.
$$

This is excellent for transparent validation at small swarm size but unsuitable as a direct large-swarm safety architecture.

## 11. The learned model imitates a classical target

The neural network learns the classical frontier utility. It does not discover a fundamentally new long-horizon objective.

Its excellent regression metrics therefore demonstrate target approximation rather than independent optimality.

## 12. Closed-loop learning results are scenario-dependent

The fully Learned controller did not consistently outperform Classical. The Hybrid result is promising in the canonical benchmark, but it is based on five paired seeds and a lightweight scenario family. It should not be interpreted as broad statistical proof that the Hybrid policy is universally superior.

## 13. The Hybrid controller rarely overrides Classical

Only 4 of 856 canonical Hybrid decisions were genuine learned overrides. Most learned-authorized decisions were agreements.

This means the final architecture is best described as a **classical planner with a learned advisor and bounded deviation**, not a predominantly neural planner.

## 14. The fault model is simplified

Failure is a hard permanent vehicle failure. Real systems may experience partial actuator faults, sensor degradation, intermittent power, navigation failure, communication-only failure, or Byzantine/incorrect state reports.

## 15. The visualization is omniscient by design

The GIFs show global information so a human can understand the mission. The display should never be mistaken for an individual UUV's observation.

## 16. Exact historical numerical reproduction is incomplete

The authoritative notebook run generated the canonical artifacts, but some early stochastic states were not fully checkpointed. Fresh modular executions can therefore produce different exact metrics while preserving the validated algorithms and safety/information invariants.

## 17. No ROS 2, Gazebo, or hardware deployment is claimed

The current repository is an autonomy research prototype. It does not demonstrate real UUV deployment, real acoustic networking, hardware-in-the-loop operation, or certified operational safety.

---

# Probable Extensions

The next stages should increase realism only when they answer a concrete research question.

## Near-term: stronger experimental reproducibility

- checkpoint every random generator and mission state required for bit-for-bit artifact regeneration;
- add deterministic scenario manifests;
- export per-seed controller results alongside aggregate tables;
- increase the number of paired benchmark seeds;
- add confidence intervals and significance tests;
- benchmark across multiple seabed images and resource distributions;
- retain machine-readable experiment configuration with every artifact bundle.

## Near-term: richer communication research

- bandwidth-limited messaging;
- message prioritization;
- event-triggered communication;
- learned communication value;
- age-of-information metrics;
- stale-map conflict resolution;
- explicit message-size accounting;
- network partitions and delayed reconnection;
- multi-hop routing over multiple rounds without violating local information causality.

## Near-term: stronger distributed allocation

- consensus-based bundle allocation;
- auction-style bidding with asynchronous delivery;
- task leases and ownership expiry;
- formal duplicate-assignment prevention;
- workload-balancing objectives;
- heterogeneous UUV capability models;
- resource-specific revisit or inspection tasks.

## Near-term: more meaningful learning targets

Instead of learning the immediate classical frontier score, future learned components could estimate:

- expected long-horizon coverage gain;
- expected redundancy avoided;
- probability of discovering a resource;
- communication value of reaching a connectivity region;
- recovery value under predicted vehicle risk;
- task completion time;
- multi-agent marginal utility;
- value of information under intermittent communication.

This would allow the neural component to contribute knowledge that is not simply an imitation of the existing heuristic.

## Medium-term: scalable safety

For larger swarms, replace or augment exhaustive $5^N$ search with methods such as:

- decentralized control barrier functions;
- local MPC;
- velocity-obstacle methods;
- priority-based reservation;
- graph-local safety optimization;
- hierarchical safety filters.

The current exhaustive shield can remain a small-$N$ reference implementation for validating those approximations.

## Medium-term: uncertainty

Introduce controlled uncertainty one source at a time:

- noisy resource detection;
- false positives and false negatives;
- localization drift;
- uncertain peer positions;
- delayed state estimates;
- map-registration error;
- probabilistic occupancy or traversability.

This would move the problem from deterministic partial information toward uncertain decentralized belief management.

## Medium-term: richer fault models

Add:

- intermittent vehicle failure;
- degraded speed;
- sensor loss;
- communication-only failure;
- partial actuator failure;
- false heartbeat suspicion;
- recovery from a false-positive failure classification;
- multiple simultaneous failures;
- graceful mission reconfiguration under reduced swarm size.

## Medium-term: heterogeneous teams

Different UUVs could have different:

- sensing footprints;
- speeds;
- communication ranges;
- endurance budgets;
- resource-classification capabilities;
- payload roles.

This would turn task allocation from geometric assignment into capability-aware distributed mission planning.

## Long-term: calibrated marine simulation

The lightweight autonomy stack can eventually be moved into higher-fidelity environments with:

- metric world coordinates;
- bathymetry;
- 3D motion;
- heading dynamics;
- currents;
- buoyancy and drag;
- thruster dynamics;
- sonar or optical sensing models;
- realistic acoustic communications.

## Long-term: ROS 2 and Gazebo

A natural transition is to preserve the decentralized autonomy interfaces while replacing the lightweight simulator boundary with ROS 2 topics/services/actions and a higher-fidelity simulation backend.

Potential components include:

```text
UUV state estimator
       ↓
local perception node
       ↓
private map / knowledge node
       ↓
frontier planner
       ↓
communication manager
       ↓
distributed allocation
       ↓
safety filter
       ↓
vehicle controller
```

The critical invariant should survive the transition:

$$
K_i(t)\neq K_{world}.
$$

## Long-term: physical UUV research

Only after the algorithms survive higher-fidelity simulation should the project claim relevance to hardware experiments involving real sensors, real localization error, acoustic modems, timing constraints, and vehicle dynamics.

The intended fidelity ladder is therefore:

```text
Lightweight 2D autonomy sandbox
            ↓
Validated decentralized algorithms
            ↓
Expanded statistical benchmarks
            ↓
ROS 2 interfaces
            ↓
Higher-fidelity marine simulation
            ↓
Hardware-in-the-loop
            ↓
Controlled physical UUV experiments
```

---

# Project Status

**Prototype status:** `validated_research_prototype`

Validated core configuration:

| Item | Value |
| --- | ---: |
| UUV agents | 4 |
| Seabed dimensions | 1448 x 1086 px |
| Local FOV | 128 x 128 px |
| Operational motion step | 32 px |
| Safe separation | 128 px |
| Communication range | 400 px |
| Packet-loss probability | 0.25 |
| Communication delay | 1 to 5 steps |
| Blackout-start probability | 0.02 |
| Frontier model | 6 -> 32 -> 32 -> 1 MLP |
| Hybrid learned margin gate | 0.03 |
| Maximum accepted classical regret | 0.02 |
| Controller benchmark | 5 paired seeds x 3 controllers x 180 steps |
| Fault showcase | 260 steps, UUV-1 failure at step 70 |

---

# License

This repository is distributed under the **Peacock Dynamics Proprietary License**.

Public visibility of the repository does not constitute an open-source release or grant a general license to use, modify, redistribute, deploy, commercialize, or create derivative works from the project.

See:

```text
LICENSE
```

for the complete terms.

---

## Peacock Dynamics

**MultiUUV-Link** extends the Peacock Dynamics lightweight-autonomy approach from single-agent seabed exploration toward communication-aware, failure-resilient, decentralized multi-robot autonomy.

The project intentionally begins with a small abstraction so that the difficult systems questions remain visible:

> **What does each robot know, what can it communicate, what work should it do, how does it remain safe, and what happens when part of the team disappears?**

---
