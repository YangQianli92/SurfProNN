# SurfProNN

SurfPro-NN:A 3D Point Cloud Neural Network for the scoring of protein–protein docking models based on Surfaces features and Protein language models

# Abstract

Protein-protein interactions (PPI) play a crucial role in numerous key biological processes, and the structure of protein complexes provides valuable clues for in-depth exploration of molecular-level biological processes. Protein-protein docking technology is widely used to simulate the spatial structure of proteins. However, there are still challenges in selecting candidate conformations (decoys) that closely resemble the native structure from protein-protein docking simulations.
In this study, we introduce a docking evaluation method based on three-dimensional point cloud neural networks named SurfPro-NN,which represents protein structures as point clouds and learns interaction information from protein interfaces by applying a state-of-the-art point cloud network architecture. With the continuous advancement of deep learning in the field of biology, a series of knowledge-rich pre-trained models have emerged. We incorporate protein surface representation models and language models into our approach, greatly enhancing feature representation capabilities and achieving superior performance in protein docking model scoring tasks.
Through comprehensive testing on public datasets, we find that our method outperforms state-of-the-art deep learning-based approaches in protein-protein docking model scoring. Not only does it significantly improve performance, but it also greatly accelerates training speed. This study demonstrates the potential of our approach in addressing protein interaction assessment problems, providing strong support for future research and applications in the field of biology.

<p align="center">
  <img src="graph abstract.png" alt="protocol" width="50%">
</p> 

# Usage
We provide a pre-trained model for inference use, there are two samples in the data directory, which are PLM features as well as the features involved in the paper, you can refer to the method in the paper to build the features, here due to the limitation, we can't upload the complete data for use, the inference code is used as follows  
``python inference.py``
