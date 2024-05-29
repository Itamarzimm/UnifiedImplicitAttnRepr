<div align="center">
<h1> A Unified Implicit Attention Formulation for Gated-Linear Recurrent Sequence Models </h1>
Itamar Zimerman<sup>1</sup> *, Ameen Ali<sup>1</sup> * and Lior Wolf<sup>1</sup>
<br>
itamarzimm@gmail.com, ameenali023@gmail.com, liorwolf@gmail.com 
<br>
<sup>1</sup>  Tel Aviv University,  
(*) equal contribution
</div>

<br>
<br>

This repository provides the official implementation for [A Unified Implicit Attention Formulation for Gated-Linear Recurrent Sequence](https://arxiv.org/abs/2405.16504). 

The purpose of this repository is to provide tools for the explainability and interpretability of modern sub-quadratic architectures, based on implicit attention representation.


<div>
    <h3> Supported Models:</h3>
    <ul class="no-bullets">
        <li><a href="https://arxiv.org/abs/2312.00752">Mamba</a></li>
        <li><a href="https://arxiv.org/abs/2402.19427">Griffin</a></li>
        <li><a href="https://arxiv.org/abs/2305.13048">RWKV</a></li>
        <li><a href="https://arxiv.org/abs/2401.09417">Vision Mamba</a></li>
    </ul>
</div>
<br>
<br>
<center>
<div>
<img src="assets/MainFig.jpg" alt="Left Image" width="500" height="200">
<br>
<img src="assets/AttnMatandXAI.jpg" alt="Left Image" width="500" height="200">
</div>
</center>



## Usage:
We provide the following Jupyter notebooks ('I' denotes installation instructions.):
- RWKV
[Notebook](https://github.com/Itamarzimm/UnifiedImplicitAttnRepr/blob/main/HF/RWKVImplicitAttnDemo.ipynb) , [I](RWKV&GriffinInstall.md)
 - Griffin 
[Notebook](https://github.com/Itamarzimm/UnifiedImplicitAttnRepr/blob/main/HF/GriffinImplicitAttnDemo.ipynb), [I](RWKV&GriffinInstall.md)
 - Mamba [Notebook](https://github.com/Itamarzimm/UnifiedImplicitAttnRepr/blob/main/MambaNLP/MambaIpmlicitAttnDemo.ipynb), [I](MambaNLPInstall.md)
 - Vision Mamba (Coming Soon!) 
 - * Heatmaps Extraction (Coming Soon!)
   * Segmentation (Coming Soon!)

## Citation
If you use this codebase, or otherwise found our work valuable, please cite:
```latex
@misc{zimerman2024unified,
      title={A Unified Implicit Attention Formulation for Gated-Linear Recurrent Sequence Models}, 
      author={Itamar Zimerman and Ameen Ali and Lior Wolf},
      year={2024},
      eprint={2405.16504},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgement:
This repository is heavily based on [Transformers](https://github.com/huggingface/transformers) and [Mamba](https://github.com/state-spaces/mamba). Thanks for their wonderful works.
