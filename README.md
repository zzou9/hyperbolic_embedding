# Hyperbolic Origami 2-Torus Visualizer

This repository contains visualization programs for two approximating embeddings of **hyperbolic 2-tori** in the **Klein model** of hyperbolic 3-space.  
These results accompany the paper [*Vertex-minimal hyperbolic origami 2-torus*](https://arxiv.org/abs/2509.18668).

---

## Overview

- **10-vertex model** – Derived from one of the 865 isomorphism classes of combinatorial vertex-minimal triangulations.  
- **12-vertex model** – Constructed from a 7-regular triangulation of a genus-2 surface.  
  This model exhibits a **2-fold rotational symmetry** about the *z-axis*, corresponding to the **hyperelliptic involution**.  
  The 12 vertices can be grouped into six pairs; the involution acts by transposing each pair and fixes the midpoint of the connecting edge.

---

## Repository Contents

### Mathematica Visualizations

#### `2-torus-10-vertices.nb`
Visualizes the **10-vertex embedding** of the 2-torus in the Klein model.  

#### `2-torus-12-vertices.nb`
Visualizes the **12-vertex embedding** of the 2-torus in the Klein model.  

---

### Python Visualization Tools

#### `10-vertex-slicer.py`
An interactive cross-section viewer for the 10-vertex embedding.  
The program uses **Matplotlib** to display slices of the model along any user-defined plane.

**Requirements**
```bash
python >= 3.8
numpy == 1.23.0
matplotlib == 3.6.0
```

**Usage**
```bash
python 10-vertex-slicer.py
```

---

#### `12-vertex-slicer.py`
An interactive cross-section viewer for the 12-vertex embedding.  
Functions identically to the 10-vertex version but loads the 12-vertex model.

**Requirements**
```bash
python >= 3.8
numpy
matplotlib
```

**Usage**
```bash
python 12-vertex-slicer.py
```

---

## Citation

If you use this code or visualizations in your work, please cite the associated paper:

> Z. Zou, “Vertex-minimal hyperbolic origami 2-torus,” *arXiv preprint* [arXiv:2509.18668](https://arxiv.org/abs/2509.18668), 2025.

---

## License

This project is released under the **MIT License**.  
See [`LICENSE`](./LICENSE) for details.
