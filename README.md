
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StefanDenn3r/Unsupervised_Anomaly_Detection_Brain_MRI/blob/master/Unsupervised%20Anomaly%20Detection%20Brain-MRI.ipynb)
[![DOI:10.1016/j.media.2020.101952](https://zenodo.org/badge/DOI/10.1016/j.media.2020.101952.svg)](https://doi.org/10.1016/j.media.2020.101952)
[![DOI:10.48550/arXiv.2004.03271](https://zenodo.org/badge/DOI/10.48550/arXiv.2004.03271.svg)](http://arxiv.org/abs/2004.03271)


# Autoencoders for Unsupervised Anomaly Segmentation in Brain MR Images: A Comparative Study

This repository contains the code for our paper on [Autoencoders for Unsupervised Anomaly Segmentation in Brain MR Images: A Comparative Study](https://www.sciencedirect.com/science/article/abs/pii/S1361841520303169). 
If you use any of our code, please cite:
```
@article{Baur2020,
  title = {Autoencoders for Unsupervised Anomaly Segmentation in Brain MR Images: A Comparative Study},
  author = {Baur, Christoph and Denner, Stefan and Wiestler, Benedikt and Albarqouni, Shadi and Navab, Nassir},
  url = {http://arxiv.org/abs/2004.03271},
  year = {2020}
}

```
```
@article{baur2021autoencoders,
  title={Autoencoders for unsupervised anomaly segmentation in brain mr images: A comparative study},
  author={Baur, Christoph and Denner, Stefan and Wiestler, Benedikt and Navab, Nassir and Albarqouni, Shadi},
  journal={Medical Image Analysis},
  pages={101952},
  year={2021},
  publisher={Elsevier}
}
```
* [Autoencoders for Unsupervised Anomaly Segmentation in Brain MR Images: A Comparative Study](#autoencoders-for-unsupervised-anomaly-segmentation-in-brain-mr-images-a-comparative-study)
  * [Requirements](#requirements)
  * [Folder Structure](#folder-structure)
  * [Usage](#usage)
      * [Config file format](#config-file-format)
      * [CLI-Usage](#cli-usage)
  * [Disclaimer](#disclaimer)
  * [License](#license)
    

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.6

All packages used in this repository are listed in [requirements.txt](https://github.com/StefanDenn3r/Unsupervised_Anomaly_Detection_Brain_MRI/blob/master/requirements.txt).
To install those, run `pip3 install -r requirements.txt`


## Folder Structure
  ```
  Unsupervised_Anomaly_Detection_Brain_MRI/
  │
  ├── Unsupervised Anomaly Detection Brain-MRI.ipynb - Jupyter notebook to work on Google Colab
  ├── run.py - execute to run in commandline
  ├── config.json - holds configuration
  │
  ├── data_loaders/ - Definition of dataloaders
  │   ├── BRAINWEB.py
  │   ├── MSISBI2015.py
  │   └── ...
  │
  ├── logs/ - default directory for storing tensorboard logs
  │
  ├── mains/ - Main files to train each architecture
  │   ├── main_AE.py
  │   └── ...
  │
  ├── model/ - Architecture definitions
  │   ├── autoencoder.py
  │   └── ...
  │
  ├── trainers/ - trainers including definition of loss functions, metrics and restoration methods
  │   ├── AE.py
  │   └── ...
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```

## Usage

Since we utilized a private dataset for training on healthy data we exchanged this dataset in the code with the publicly available Brainweb dataset. 
The Brainweb dataset can be downloaded [here](https://brainweb.bic.mni.mcgill.ca/). For easy use, we also provide a script, which allows you to download the dataset. The script can be found in [here](https://github.com/StefanDenn3r/Unsupervised_Anomaly_Detection_Brain_MRI/blob/master/utils/brainweb_download.py). 

### Config file format
First define the path to the data directories in `config.default.json`.
Of course only those you want to use have to be defined. 
If you want to use your own dataset, check how the dataloaders in `dataloaders` 
are defined and implement your own to work with our code.
```json
{
  "BRAINWEBDIR": "path/to/Brainweb",
  "MSSEG2008DIR": "path/to/MSSEG2008",
  "MSISBI2015DIR": "path/to/ISBIMSlesionChallenge",
  "MSLUBDIR": "path/to/MSlub",
  "CHECKPOINTDIR": "path/to/saved/checkpoints",
  "SAMPLEDIR": "path/to/saved/sample_dir"
}
```

### CLI Usage
For the results of our paper we used the `run.py`. 
Every model can also be trained individually using the script which are provided in the `mains` folder.


### Google Colab Usage
Offering you an easy start to work with the code base is essential. Therefore, we prepared a Jupyter Notebook for Google Colab, which can be found here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StefanDenn3r/Unsupervised_Anomaly_Detection_Brain_MRI/blob/master/Unsupervised%20Anomaly%20Detection%20Brain-MRI.ipynb)
Running all cells will download the brainweb dataset, train all models and evaluate them. Enjoy! 

## Disclaimer
The code has been cleaned and polished for the sake of clarity and reproducibility, and even though it has been checked thoroughly, it might contain bugs or mistakes. Please do not hesitate to open an issue or contact the authors to inform of any problem you may find within this repository.

## License
This project is licensed under the GNU General Public License v3.0. See LICENSE for more details
