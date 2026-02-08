# Quantum Entanglement Detection in Bipartite Qutrit Systems using Machine Learning

## Overview
This project develops a symmetry-respecting machine-learning framework for detecting quantum entanglement in bipartite qutrit (3 × 3) systems.  
Unlike qubit systems, qutrit states inhabit a higher-dimensional Hilbert space, making entanglement detection both theoretically subtle and computationally demanding.

Rather than optimizing raw classification accuracy, this work focuses on constructing a physically consistent dataset, eliminating spurious learning shortcuts, and studying the intrinsic limits of entanglement detection using local observables.

---

## Scientific Background
Quantum entanglement is a central resource in quantum information science, with applications in communication, cryptography, and computation.  
Determining whether a given quantum state is entangled becomes increasingly challenging in higher-dimensional systems.

For bipartite qutrit states, the **Positive Partial Transpose (PPT) criterion** provides a necessary and sufficient condition for separability in many relevant cases. While exact PPT tests are computationally feasible for individual states, large-scale studies motivate data-driven approaches—provided physical symmetries and constraints are properly enforced.

---

## Dataset Construction
Quantum states are generated using a **graph-based construction** of mixed bipartite qutrit density matrices:

- Random weighted graphs are generated with a fixed tensor-product labeling
- Adjacency matrices are mapped to Hermitian, positive semidefinite density matrices with unit trace
- States are labeled as **entangled or separable** using the PPT criterion
- Sampling is performed **near the entanglement–separability boundary** to avoid trivial classification
- Local-unitary (LU) symmetry is enforced through data augmentation

This procedure eliminates dataset biases that can artificially inflate machine-learning performance.

---

## Feature Representation
Each quantum state is represented using expectation values of tensor products of **Gell–Mann matrices**, yielding a complete set of local correlators for two-qutrit systems.  
This representation captures physically meaningful correlations while remaining invariant under local basis changes when combined with LU augmentation.

---

## Methodology
The following supervised learning models were evaluated:
- Gradient Boosting
- Random Forests
- Support Vector Machines
- Regularized Multilayer Perceptrons (MLPs)

Model selection and tuning were guided by the **geometry of the entanglement boundary**, not raw accuracy alone.  
Validation-aware threshold optimization and ensemble methods were employed to ensure fair comparison.

---

## Results
After enforcing physical constraints and removing dataset bias:
- A **regularized MLP ensemble** achieved the best performance
- Test accuracy saturated at approximately **92%**
- ROC–AUC reached **≈ 0.98**
- Further performance gains were limited by the intrinsic indistinguishability of weakly entangled and separable states near the PPT boundary

These results indicate a **fundamental performance ceiling** when using local correlator information alone.


---

## Key Insight
This work demonstrates that naïve machine-learning pipelines can significantly overestimate entanglement-detection performance.  
When physical symmetries and boundary effects are properly enforced, classification accuracy is fundamentally limited—revealing an information-theoretic constraint rather than a modeling deficiency.

---

## Status
This project is intended as a research study and first publication effort, with ongoing work focused on visualization, boundary-resolved analysis, and quantitative correlations between classifier confidence and entanglement strength.
