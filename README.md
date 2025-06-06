# On the Importance of Text Preprocessing for Multimodal Representation Learning and Pathology Report Generation

This repository contains all code to support the paper:

***"On the Importance of Text Preprocessing for Multimodal Representation Learning and Pathology Report Generation"***

[[`arXiv`](https://arxiv.org/pdf/2502.19285)]

<div align="center">
  <img width="100%" alt="Overview" src=".github/model_overview.png">
</div>

## Model Parameters

We provide checkpoints for both the retrieval and report generation stages. All models are available on Hugging Face.

### 🔍 Stage 1: Retrieval Model

The retrieval model is trained with 16 queries and is used for the retrieval results presented in the paper.

- [**Retrieval Model**](https://huggingface.co/RTLucassen/@TODO)

### 📝 Stage 2: Report Generation Models

The final report generation models build upon the Stage 1 checkpoint trained with 64 queries and are used for the reader study results.

- [**Base Checkpoint**](https://huggingface.co/RTLucassen/@TODO)

Final Stage 2 models:

- [**Model – Full Report**](https://huggingface.co/RTLucassen/@TODO)  
- [**Model – H&E-only**](https://huggingface.co/RTLucassen/@TODO)

## Citing

If you found our work useful in your research, please consider citing our paper:

```bibtex
@misc{lucassen2025importancetextpreprocessingmultimodal,
  title={On the Importance of Text Preprocessing for Multimodal Representation Learning and Pathology Report Generation}, 
  author={Ruben T. Lucassen and Tijn van de Luijtgaarden and Sander P. J. Moonemans and Gerben E. Breimer and Willeke A. M. Blokx and Mitko Veta},
  year={2025},
  eprint={2502.19285},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2502.19285}
}
```