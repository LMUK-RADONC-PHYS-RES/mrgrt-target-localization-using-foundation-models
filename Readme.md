# MRgRT target tracking using foundation models

This repository contains the source code for the models presented in the paper [MRgRT real-time target localization using foundation models for contour point tracking and promptable mask refinement](URL).

It can also be used as a starting point for similar projects.

## Data

This repository does not contain any 2D+t CineMRI data. For example purposes the `generate_example_sequence.py` script can generate a (noisy) image sequence of a circle oscillating up and down.

Interested readers are encoraged to implement data loading routines for their own specific dataformats.

### Weights

The weights used for the paper mentioned above were obtained from the following urls. Running the following command should place them in the correct positions, as expected by the source code.

```bash
cd checkpoints

curl -O https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt
curl -O https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
# other sam2 variants you might want to try
#curl -O https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
#curl -O https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt

curl -L -O https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth

curl -L -O https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_sam.pt

# verify sha256 checksums
echo "95949964d4e548409021d47b22712d5f1abf2564cc0c3c765ba599a24ac7dce3  sam2_hiera_small.pt" | sha256sum -c 
echo "7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b  sam2_hiera_large.pt" | sha256sum -c 
echo "362f5274376d610dc987b6daf2c2fefe63e06e1835f4ec1a10d0a15c5a4eef4f  cotracker2.pth" | sha256sum -c 
echo "1fd2867cf669e6d89b1c91d0eb883d83ef4bcc590143dd2ca35e2705b21a4f2f  repvit_sam.pt" | sha256sum -c 
```

## Requirements

This project relies on the dependencies listed in the `requirements.txt` file, which can be installed with the following command:

```bash
pip install -r requirements.txt
```

Additionally, the project makes use of a number of open source models, for which the source code was directly copied into this repository. See the [License](#license) section of this readme for more information.


## Citation

Please cite the following paper when making use of the presented code:

```bibtex
@article{Bloecker2024,
  title = {MRgRT real-time target localization using foundation models for contour point tracking and promptable mask refinement},
  ISSN = {1361-6560},
  url = {http://dx.doi.org/10.1088/1361-6560/ad9dad},
  DOI = {10.1088/1361-6560/ad9dad},
  journal = {Physics in Medicine &amp; Biology},
  publisher = {IOP Publishing},
  author = {Bl\"{o}cker,  Tom Julius and Lombardo,  Elia and Marschner,  Sebastian and Belka,  Claus and Corradini,  Stefanie and Palacios,  Miguel A and Riboldi,  Marco and Kurz,  Christopher and Landry,  Guillaume},
  year = {2024},
  month = dec 
}
```

## License

The original code in the repository is released as open source under the conditions contained in the LICENSE file (Apache License 2.0).

All files in the cotracker, sam1, and sam2, repvit subfolders of this repository are part of the respective open source software projects, according to the respective licences, contained in the respective folders.

The original repositories for can be found under the following urls:

- cotracker (co-tracker): https://github.com/facebookresearch/co-tracker
- sam1 (segment-anything): https://github.com/facebookresearch/segment-anything
- sam2 (segment-anything 2): https://github.com/facebookresearch/segment-anything-2
- repvit (repvit_sam): https://github.com/THU-MIG/RepViT

Note that this repository only contains a subset of these repositories, as required for its purposes. Interested readers are directed to the respective main repositories for more information or upstream changes.

## FAQ

### How to solve problems with pytorch versions and other dependencies?

The versions provided `requirements.txt` file were selected for the specific system used during the development leading up to the publication of the paper in the summer of 2024. For this reason there is a good chance that the your not working for your specific setup.

As a good first step to try it is recommended to remove or comment out the version freeze for the numpy package, as well as to remove of comment out the specification of the pytorch index-url for the cuda 11.8 version.

### Can I run the code without a cuda gpu?

Note that segment-anything 2 at the time of writing was only compatible with cuda, not the cpu or mps pytorch backends. For this reason the corresponding model can only be used with an cuda-enabled gpu.

The other presented contour tracking model is not limited in this regard.

### How to evaluate the model outputs?

This repository does not contain any code for the evaluation of predictions produced by the model. Interested readers may implement or import their own evaluation/metrics.

### How to contribute?

This repository is not designed as an ongoing effort, but as a one-time publication.