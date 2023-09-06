# scFoundation
**Large Scale Foundation Model on Single-cell Transcriptomics**

Large-scale pretrained models have become foundation models leading to breakthroughs in natural language processing and related fields. Developing foundation models in life science for deciphering the "languages" of cells and facilitating biomedical research is promising yet challenging. We developed a large-scale pretrained model scFoundation with 100M parameters for this purpose. scFoundation was based on the **xTrimoGene** architecture and trained on over 50 million human single-cell transcriptomics data, which contain high-throughput observations on the complex molecular features in all known types of cells. scFoundation is currently the largest model in terms of the size of trainable parameters, dimensionality of genes and the number of cells used in the pre-training. Experiments showed that scFoundation can serve as a foundation model for single-cell transcriptomics and achieve state-of-the-art performances in a diverse array of downstream tasks, such as gene expression enhancement, tissue drug response prediction, single-cell drug response classification, and single-cell perturbation prediction. More information can be obtained at https://www.biorxiv.org/content/10.1101/2023.05.29.542705 .

## API
We are excited to announce the availability of our API for cell and gene embedding inference. To get started:

1. **Register and Agreement Acknowledgment**: Visit https://api.biomap.com/xTrimoGene/apply to register account, sign the [Terms](https://uc.biomap.com/#/protocal-terms) and apply for your API token.
2. **Review Process**: Upon receiving your application, our team will evaluate your application.
3. **Token Issuance**: If your application is approved, you will receive an email containing your API token along with guidelines and restrictions on our API usage.
4. **Getting Started with the API**: For instructions on using the API, navigate to the `apiexample` directory in our repository.

## Pre-training
We now provide API to obtain the cell embeddings from scFoundation. The embeddings required for most downstream tasks shown in our manuscript have been provided in this repository. 

### packages for pre-training
The pre-training framework is implemented in PyTorch and MMF. Given a stable internet connection, it will take several minutes to install these packages.
```
torch==1.10.0
mmf=1.0.0rc12
```
For the downstream tasks, the required packages are listed in their repository.

## For downstream task
This repository provides the source code necessary to use the scFoundation generated cell and gene embeddings for several downstream tasks such as gene expression enhancement, drug response prediction and perturbation prediction. The source codes for the downstream tasks are in the following repositories:

### Read depth enhancement
The results of SAVER, scImpute, MAGIC were obtained from the SAVER repository (https://github.com/mohuangx/SAVER-paper). The results of scFoundation were obtained by running the bash `run.sh` in the `enhancement` folder. You can find details in the `enhancement/README.md`.

### DeepCDR
The baseline code is from https://github.com/kimmo1019/DeepCDR

Please follow the commands in `DeepCDR/prog/run.sh`. The scFoundation embeddings of Bulk data are at `DeepCDR/data/50M-0.1B-res_embedding.npy`. You can find details in the `DeepCDR/README.md`.

### SCAD
The baseline code is from https://github.com/CompBioT/SCAD

Please follow the steps detailed in `SCAD/README.md`. The scFoundation embeddings of Bulk and single cell data are in the `SCAD/data/split_norm/` folder.

### GEARS
The baseline code is from https://github.com/snap-stanford/GEARS.

The commands required for running the code can be found in  `GEARS/run_sh`. The gene embedding of each cell is 19264*512 which is too large to be saved. We generated the gene embedding during the training process. You can find details in the `GEARS/README.md`.

### Gene module inference
In this directory, you'll find the code necessary for inferring gene modules from gene context embeddings.

### Cell mapping
This directory contains the demo usage code and scripts to reproduce figures related to the cell mapping task.

### Cell type annotation
Here, you'll find the code to reproduce results for the cell type annotation task.

## Copyright Notice
### Code License

Source code is licensed under the permissive Apache Licence, Version 2.0.

### Third-party Software License

Use of the third-party software, libraries or code referred to in the Acknowledgements section may be governed by separate terms and conditions or license provisions.

Your use of the third-party software, libraries or code is subject to any such terms and you should check that you can comply with any applicable restrictions or terms and conditions before use.

## Reference

- [scvi-tools](https://github.com/scverse/scvi-tools)
- [SAVER](https://github.com/mohuangx/SAVER)
- [DeepCDR](https://github.com/kimmo1019/DeepCDR)
- [SCAD](https://github.com/CompBioT/SCAD)
- [GEARS](https://github.com/snap-stanford/GEARS)

## Acknowledgements

scFoundation uses and/or references the following separate libraries and packages (ordered alphabetically):

- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [Docker](https://www.docker.com/)
- [einops](https://github.com/arogozhnikov/einops)
- [MMF](https://github.com/facebookresearch/mmf)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Pytorch](https://pytorch.org/)
- [Pytorch Lightning](https://www.pytorchlightning.ai)
- [PyYAML](https://pyyaml.org)
- [Scipy](https://scipy.org/)
- [Tqdm](https://github.com/tqdm/tqdm)

Thanks for all their contributors and maintainers!