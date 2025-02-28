<p align="center">
    <img src="docs/static/DSG_LOGO_REDESIGN.png" alt="DSG logo" width="200"/>
</p>

# SatVision-TOA: A Geospatial Foundation Model for All-Sky Remote Sensing Imagery

[![DOI](https://zenodo.org/badge/472450059.svg)](https://zenodo.org/badge/latestdoi/472450059)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://github.com/nasa-nccs-hpda/pytorch-caney/blob/docs/README.md)
[![Paper](https://img.shields.io/badge/arXiv-2411.17000-blue)](https://arxiv.org/abs/2411.17000)

Implementation of the SatVision-TOA model for All-Sky Remote Sensing Imagery.

_The package includes the pretrained SatVision-TOA model along with fine-tuned versions for both 1) Image Reconstruction and 2) 3D Cloud Retrieval._
_While instructions for training the model are provided for reference, this deployment is primarily intended for downstream SatVision-TOA task usage and customization._

Concept and design details are provided in the paper on [arXiv.](https://arxiv.org/pdf/2411.17000)

See system requirements for [installation instructions](requirements/README.md) 

[Please see the documentation for detailed instructions and more examples.](https://microsoft.github.io/satvision)
You can also directly go to [a full-fledged example that runs the model on ERA5](https://microsoft.github.io/satvision/example_era5.html).


Contents:

- [What is SatVision-TOA?](#what-is-satvision)
- [Why SatVision-TOA?](#why-satvision)
- [Getting Started](#getting-started)
- [API](#api)
- [Contributing](#contributing)
- [License](#license)
- [FAQ](#faq)
- [Citation](#citation)

## What is SatVision-TOA?
We introduce SatVision-TOA, a novel foundation model pre-trained on 14-band MODIS L1B Top-Of-Atmosphere (TOA) radiance imageries, addressing the need for models that handle moderate- and coarse-resolution all- sky remote sensing data. The SatVision-TOA model is pre-trained using a Masked-Image-Modeling (MIM) framework and the SwinV2 architecture, and learns detailed contextual representations through self-supervised learning without the need for labels. It is a 3 billion parameter model that is trained on 100 million images.  To validate the model we focused on two key applications: 
* 1) image reconstruction performance, which evaluates the masked-image-modeling task on unseen data, 
* 2) and 3D cloud retrieval, a downstream task targeting the prediction of vertical cloud structure.

## Why SatVision-TOA?
To our knowledge this is the largest foundation model trained solely on satellite remote sensing imagery. Initial results indicate that SatVision-TOA achieves superior performance over baseline methods when applied to downstream tasks such as 3D cloud retrieval.  Our work advances pre-trained vision modeling for multispectral remote sensing by learning from a variety of atmospheric and aerosol conditions to improve cloud and land surface monitoring.

_SatVision-TOA is a _foundation model_, which means that it was first generally trained on a lot of data,_
_and then can be adapted to specialised cloud and land surface monitoring tasks with relatively little data._

## Getting Started

See requirements and installation [instructions](requirements/README.md) 

## API

See requirements and installation [instructions](requirements/README.md) 

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

See [`LICENSE.txt`](LICENSE.txt).

### Use of this code
Our goal in publishing this code is
(1) to facilitate reproducibility of our paper and
(2) to support and accelerate further research into foundation model for atmospheric forecasting.
This code has not been developed nor tested for non-academic purposes and hence should not be used as such.

### Limitations
Although  was trained to accurately predict future weather, air pollution, and ocean waves,
SatVision-TOA is based on neural networks, which means that there are no strict guarantees that predictions will always be accurate.
Altering the inputs, providing a sample that was not in the training set,
or even providing a sample that was in the training set but is simply unlucky may result in arbitrarily poor predictions.
In addition, even though SatVision-TOA was trained on a wide variety of data sets,
it is possible that SatVision-TOA inherits biases present in any one of those data sets.
A forecasting system like SatVision-TOA is only one piece of the puzzle in a weather prediction pipeline,
and its outputs are not meant to be directly used by people or businesses to plan their operations.
A series of additional verification tests are needed before it can become operationally useful.

### Data
The models included in the code have been trained on a variety of publicly available data.
A description of all data, including download links, can be found in [Supplementary C of the paper](https://arxiv.org/pdf/2405.13063).
The checkpoints include data from ERA5, CMCC, IFS-HR, HRES T0, GFS T0 analysis, and GFS forecasts.

### Evaluations
All versions of SatVision-TOA were extensively evaluated by evaluating predictions on data not seen during training.
These evaluations not only compare measures of accuracy, such as the root mean square error and anomaly correlation coefficient,
but also look at the behaviour in extreme situations, like extreme heat and cold, and rare events, like Storm Ciarán in 2023.
These evaluations are the main topic of [the paper](https://arxiv.org/pdf/2405.13063).

*Note: The documentation included in this file is for informational purposes only and is not intended to supersede the applicable license terms.*


## FAQ

### How do I setup the repo for local development?

First, install the repository in editable mode and setup `pre-commit`:

```bash
make install
```

To run the tests and print coverage, run

```bash
make test
```

You can then explore the coverage in the browser by opening `htmlcov/index.html`.

To locally build the documentation, run

```bash
make docs
```

To locally view the documentation, open `docs/_build/index.html` in your browser.

### Why are the fine-tuned versions of SatVision-TOA for air quality and ocean wave forecasting missing?

The package currently includes the pretrained model and the fine-tuned version for high-resolution weather forecasting.
We are working on the fine-tuned versions for air pollution and ocean wave forecasting, which will be included in due time.

## Citation

Cite us as follows:

```
@misc{satvision-base,
    author       = {Carroll, Mark and Li, Jian and Spradlin, Caleb and Caraballo-Vega, Jordan},
    doi          = {10.57967/hf/1017},
    month        = aug,
    title        = {{satvision-base}},
    url          = {https://huggingface.co/nasa-cisto-data-science-group/satvision-base},
    repository-code = {https://github.com/nasa-nccs-hpda/pytorch-caney}
    year         = {2023}
}
```
