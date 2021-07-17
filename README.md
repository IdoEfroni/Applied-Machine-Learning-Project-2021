# Applied-Machine-Learning-Project-2021

## Installation
* To activate the different algorithms and get the results you must run the **project.py**.
* Install the requirements of the project (pip install requirements.txt)
* Run pip install git+http://github.com/nikitadurasov/masksembles
* **NOTICE: you may need to write pip3 instead of pip**
## The Data
  * **NOTICE: The data is too heavy, so we can't upload all of the datasets to Github, further instruction are mentioned below**
  * Before running project.py it requires **filling the Csv_Data folder with valid data**.
  * To fill the folder we created a **Colab Notebook**, that takes **data images from Kaggle API** and performs pre-processing on the data.
  * The link is located below or here: 
    * [![Open Pre-Process Notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/157g4Gju2nhtRn-tOSIiTjFrhU8cq8_Sj?usp=sharing)
    * We have taken 5 large datasets from Kaggle API which were pre-processed to 10 different datasets (the other 10 datasets were taken from Keras API). 
    * **NOTICE: Don't forget to put the CSV that were generated in colab, into the folder Csv_Data**
 ## Running the Project (The algorithms) 
  * In order to choose the algorithm (VGG, Masksembles or the transfer learning) you need to change the flag_vgg and flag_improve
  *  **NOTICE: Lines 513-520 contain instructions**
  * After all of the pre-process is completed you can **run the project.py**.
 ## Running Statistics (Parts 5 and 6 in our project)
 * From line 288, each function has an instruction if you wish to run it. 
 * **NOTICE: it requires to uncomment the line of the function**


## link to Colab Pre-Process Notebook for the project


[![Open Pre-Process Notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/157g4Gju2nhtRn-tOSIiTjFrhU8cq8_Sj?usp=sharing)

## External Repositories

Our project is based on two main repositories:
1. Masksembles - https://github.com/nikitadurasov/masksembles
2. Bayesian Search - https://towardsdatascience.com/bayesian-hyper-parameter-optimization-neural-networks-tensorflow-facies-prediction-example-f9c48d21f795

## Citation
```
@inproceedings{Durasov21,
  author = {N. Durasov and T. Bagautdinov and P. Baque and P. Fua},
  title = {{Masksembles for Uncertainty Estimation}},
  booktitle = CVPR,
  year = 2021
}
```
