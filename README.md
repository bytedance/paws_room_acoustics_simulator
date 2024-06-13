# PAWS-Dataset

## Intro

In this project, we present a Physical Acoustic Wave Simulation (PAWS) dataset for room acoustic simulation based on wave acoustics simulation.

Previous geometrical acoustics-based room acoustic simulation methods have difficulty in modeling diffraction and low-frequency sounds. Many previous acoustic datasets only provide impulse responses at limited positions without offering the entire acoustic pressure field. There are limited acoustic field datasets available, and they mainly focus on single or a limited number of scenes. In this work, we simulate a large-scale wave acoustics-based room acoustic dataset, PAWS, using acoustic finite-difference time-domain (FDTD) and k-space solvers. The PAWS dataset contains acoustic pressure  p(x, y, z, t)  data at 65,536 grid samples in space. PAWS consists of 200 scenes randomly sampled from shoebox rooms with subrooms and obstacles, polygon rooms, and real-world scenes from the Replica dataset. We randomly place source and obstacles in each scene and run the simulation.



This is the codes for the data generation of PAWS-Dataset.  We provide example Jupyter notebook to let users make their own simulation data.

Our dataset is avaliable at <https://huggingface.co/datasets/Tamamo0v0/PAWS-Dataset/tree/main>

