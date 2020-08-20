# Welcome to OpenFL

See the documentation at: https://openfederatedlearning.readthedocs.io/en/latest/index.html

Open Federated Learning (OpenFL) is a Python3 library for federated learning. Federated learning enables organizations to collaborately train a machine learning model without sharing sensitive data with each other.

OpenFL currently supports two main types of nodes:
1. Collaborator nodes, which uses local sensitive data to train the shared (aggregated) model, then send those updates to the:
2. Aggregator node, which aggregates received model updates from collaborators and distributes the aggregated model back to the collaborators.

The aggregator is framework-agnostic, while the collaborator can use any deep learning frameworks, such as Tensorflow or Pytorch.

OpenFL is developed by the Intel Labs and the Intel Internet of Things Group.

## OpenFL and the Federated Tumor Segmentation (FeTS) Initiative

The Open Federated Learning (OpenFL) framework is developed as part of a collaboration between Intel and the University of Pennsylvania (UPenn), and describes Intel’s commitment in supporting the grant awarded to the [Center for Biomedical Image Computing and Analytics (CBICA)](https://www.cbica.upenn.edu/) at UPenn (PI: S.Bakas) from the [Informatics Technology for Cancer Research (ITCR)](https://itcr.cancer.gov/) program of the National Cancer Institute (NCI) of the National Institutes of Health (NIH), for the development of the [Federated Tumor Segmentation (FeTS, www.fets.ai)](https://www.fets.ai/) platform (grant award number: U01-CA242871). FeTS is an exciting, real-world medical FL platform, and we are honored to be collaborating with UPenn in leading a federation of international collaborators. Although OpenFL was designed to serve as the backend for the FeTS platform, and OpenFL developers and researchers continue to work very closely with UPenn on the FeTS project, OpenFL was built to be agnostic to the use-case and the machine learning framework, and we welcome input from domains outside medicine and imaging.

We’ve included the [FeTS-AI/Algorithms](https://github.com/FETS-AI/Algorithms) repository as a submodule of OpenFL to highlight how OpenFL serves as the FeTS backend. While not necessary to run OpenFL, the FeTS algorithms show real-world FL models and use cases. Additionally, the [FeTS-AI/Front-End](https://github.com/FETS-AI/Front-End) shows how UPenn and Intel have integrated UPenn’s medical AI expertise with Intel’s OpenFL to create a federated learning solution for medical imaging. 

### Requirements

- OS: Primarily tested on Ubuntu 16.04 and 18.04, but code should be OS-agnostic. (Optional shell scripts may not be).
- Python 3.5+
- Makefile setup scripts require python3.x-venv
- Sample models require TensorFlow 1.x or PyTorch. Primarily tested with TensorFlow 1.13-1.15.2 and Pytorch 1.2-1.6 

### Coming Soon
- Graphene-SGX recipes for running the aggregator inside SGX (https://github.com/oscarlab/graphene)
- Improved error messages for common errors
- FL Plan authoring guide
- Model porting guide and tutorials

Copyright (C) 2020 Intel Corporation
