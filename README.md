# Hyperbolic Origami 2-Torus Visualizer

This repository contains accompanying programs that visualizes the two approximating embeddings of hyperbolic 2-tori in the Klein model of hyperbolic 3-space. The results can be found from the [paper](https://arxiv.org/abs/2509.18668) "Vertex-minimal hyperbolic origami 2-torus."

The 10-vertex model is based on one of the 865 isomorphism classes of combinatorial vertex-minimal triangulations. The 12-vertex model is based on a 7-regular triangulation of the genus-2 surface. It has 2-fold rotational symmetry around the z-axis that is a hyperelliptic involution. One can group the 12 vertices into six pairs, and the hyperellptic involution acts by transpositions on each of the six pairs, fixing the midpoint of the edges connecting each pair of vertices. 

Below are the descriptions of the corresponding scripts:

---

### 2-torus-10-vertices.nb

A Mathematica notebook that displays the 10-vertex example in the Klein model. 

### 2-torus-12-vertices.nb

A Mathematica notebook that displays the 12-vertex example in the Klein model. 

### 10-vertex-slicer.py

A python script that implements an interactive visualization program based on python's matplotlib library. It displays a cross-section of the 10-vertex embedding with respect to any plane that the user specifies. To run the script, download to the local device and run it in the terminal with a python environment, numpy, and matplotlib installed. Then, open a terminal and run the following prompt: python 10-vertex-slicer.py

### 12-vertex-slicer.py

A python script that implements an interactive visualization program based on python's matplotlib library. It displays a cross-section of the 12-vertex embedding with respect to any plane that the user specifies. To run the script, download to the local device and run it in the terminal with a python environment, numpy, and matplotlib installed. Then, open a terminal and run the following prompt: python 10-vertex-slicer.py