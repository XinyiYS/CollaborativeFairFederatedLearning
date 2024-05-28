# Collaborative Fairness in Federated Learning

This repository is the official implementation of [Collaborative Fairness in Federated Learning
](https://arxiv.org/abs/2008.12161). **Best Paper at** [International Workshop on Federated Learning for User Privacy and Data Confidentiality
in Conjunction with IJCAI 2020](http://fl-ijcai20.federated-learning.org/) and in [Federated Learning: Privacy and Incentive](https://link.springer.com/book/10.1007/978-3-030-63076-8).

>📋 In this work, we propose a Collaborative Fair Federated Learning (CFFL) framework which modifies the existing FL paradigm in order to encourage collaborative fairness (defined in our paper) and still maintain a competitive performance.

## Citing
If you have found our work to be useful in your research, please consider citing it with the following bibtex:
```
@Inbook{Lyu2020,
    author="Lyu, Lingjuan
    and Xu, Xinyi
    and Wang, Qian
    and Yu, Han",
    title="Collaborative Fairness in Federated Learning",
    bookTitle="Federated Learning: Privacy and Incentive",
    year="2020",
    publisher="Springer International Publishing",
    address="Cham",
    pages="189--204",
}
```

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```
>📋  We recommend managing your environment using Anaconda, both for the versions of the packages used here and for easy management. 

>📋  Our code automatically detects GPU(s) through NVIDIA driver, and if not available it will use CPU instead. You can manually disable the use of GPU through changing the `arguments.py` or in the execution script `main.py` or `test.py`.



## Training

To run the code in the paper, run this command:
```
python main.py or python.test.py
```
>📋  Note that the arguments for experiments are specified within the `main.py` or `test.py` file. Alternatively, you can make changes to `arguments.py` where the default arguments are stored.


>📋  The execution script is `main.py` or `test.py`. Running `main.py` starts the full-fledged experiments and it creates and writes to corresponding directories. Running `test.py` on the other hand only executes the code without creating directories or writing to them, but prints out the message to terminal.

## Evaluation

To produce the collated accuracy and fairness results from complement execution of the code, run:

```eval
python examine_results.py --experiment_dir <experiment_dir>
```

>📋  For each experiment directory created from running `main.py`, you can run the above command to get the corresponding accuracy and fairness results, along with the figures showing the convergence behaviors.

<!---
We do not have pre-trained models currently.
## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.
--->

## Results
We expect the algorithm to perform consistently on the same machine, but might achieve slightly different results over different machines.

Our CFFL achieves the following performance on:
### [MNIST](http://yann.lecun.com/exdb/mnist/)

#### Fairness
|         |     Fedavg |     DSSGD |     CFFL |   CFFL pretrain |
|:--------|-----------:|----------:|---------:|----------------:|
| P10_0.1 | -0.504708  | -0.781777 | 0.985535 |        0.978962 |
| P10_1.0 | -0.504708  |  0.906745 | 0.987358 |        0.973035 |
| P20_0.1 |  0.604084  | -0.817737 | 0.98519  |        0.992334 |
| P20_1.0 |  0.604084  |  0.804533 | 0.985079 |        0.962809 |
| P5_0.1  |  0.0308093 |  0.907159 | 0.997595 |        0.996281 |
| P5_1.0  |  0.0308093 |  0.846134 | 0.990229 |        0.98659  |

#### Accuracy
|               |   P10_0.1 |   P10_1.0 |   P20_0.1 |   P20_1.0 |   P5_0.1 |   P5_1.0 |
|:--------------|----------:|----------:|----------:|----------:|---------:|---------:|
| Fedavg        |    0.9532 |    0.9531 |    0.9626 |    0.9626 |   0.9362 |   0.9362 |
| DSSGD         |    0.942  |    0.9011 |    0.8236 |    0.8427 |   0.9328 |   0.9156 |
| Standalone    |    0.9088 |    0.903  |    0.9064 |    0.9064 |   0.903  |   0.903  |
| CFFL          |    0.93   |    0.9424 |    0.9325 |    0.9476 |   0.9183 |   0.9261 |
| CFFL pretrain |    0.9285 |    0.9393 |    0.9334 |    0.9419 |   0.9185 |   0.9274 |


### [Adult](http://archive.ics.uci.edu/ml/datasets/Adult)

#### Fairness
|         |     Fedavg |    DSSGD |     CFFL |   CFFL pretrain |
|:--------|-----------:|---------:|---------:|----------------:|
| P10_0.1 |  0.442706  | 0.622967 | 0.920093 |        0.880041 |
| P10_1.0 |  0.442706  | 0.566592 | 0.919459 |        0.93072  |
| P20_0.1 | -0.34316   | 0.603001 | 0.805573 |        0.844079 |
| P20_1.0 | -0.34316   | 0.580096 | 0.795181 |        0.824629 |
| P5_0.1  | -0.0332574 | 0.156226 | 0.980437 |        0.985049 |
| P5_1.0  | -0.0332574 | 0.357077 | 0.993701 |        0.977457 |

#### Accuracy
|               |   P10_0.1 |   P10_1.0 |   P20_0.1 |   P20_1.0 |   P5_0.1 |   P5_1.0 |
|:--------------|----------:|----------:|----------:|----------:|---------:|---------:|
| Fedavg        |  0.831401 |  0.830062 |  0.831624 |  0.833185 | 0.825825 | 0.827609 |
| DSSGD         |  0.827832 |  0.812221 |  0.820696 |  0.793042 | 0.819358 | 0.818912 |
| Standalone    |  0.823149 |  0.814674 |  0.820696 |  0.813559 | 0.819358 | 0.812221 |
| CFFL          |  0.826271 |  0.827386 |  0.827163 |  0.828278 | 0.819581 | 0.826271 |
| CFFL pretrain |  0.826271 |  0.828055 |  0.826271 |  0.830954 | 0.818912 | 0.827832 |


>📋  Include a table of results from your paper If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋 Suggestions and questions are welcome through issues. All contributions welcome! All content in this repository is licensed under the MIT license.
