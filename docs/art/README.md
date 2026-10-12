# Art layer — TRM Lab XS

Curatorial companion for the **TRM (Two-state Recursive Model)** research repo. This is **not** a performance benchmark; it describes an artistic translation of the architecture.

## Thesis

*Thinking is traversing the same space twice—and knowing when to stop.*

## MVP piece (planned)

**Dual Orbit Loop (generative video):**

- Outer orbit → state **y** (slow cycle **T**)
- Inner orbit → state **z** (**n_inner** updates per **T**)
- Central glow → halting score (metaphor; threshold in legend)

Parameters shown on screen should match the active YAML config when exported from `experiments/visualize.py`.

## Data vs metaphor

| Element | Type |
|---------|------|
| T, n_inner from config | Documented fact |
| Orbit animation | Simulation / metaphor |
| Task accuracy in captions | Only from real `evaluate.py` logs |

## References

- Paper: [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)
- Architecture diagram: root `README.md` (Mermaid)

This page is **self-contained** for visitors; extended production notes stay with the maintainers and are not linked from here.
