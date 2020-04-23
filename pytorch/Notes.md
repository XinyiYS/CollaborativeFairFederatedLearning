
## Adult dataset dropna alpha=3

###  With gradient cut credit-sum alpha=3 
#### Fairness
	|         |   Distriubted |     CFFL |   Contributions_V_final |
	|:--------|--------------:|---------:|------------------------:|
	| P10_0.1 |     0.128077  | 0.899917 |                0.847985 |
	| P10_1.0 |    -0.165437  | 0.85923  |                0.860505 |
	| P20_0.1 |    -0.0612526 | 0.897104 |                0.868166 |
	| P20_1.0 |     0.125023  | 0.856942 |                0.839956 |
	| P5_0.1  |    -0.0786406 | 0.892601 |                0.748839 |
	| P5_1.0  |     0.360535  | 0.925036 |                0.790072 |
	Test Accuracy
	|             |   P10_0.1 |   P10_1.0 |   P20_0.1 |   P20_1.0 |   P5_0.1 |   P5_1.0 |
	|:------------|----------:|----------:|----------:|----------:|---------:|---------:|
	| Distributed |    81.253 |    81.561 |    81.41  |    81.699 |   80.794 |   80.896 |
	| Standalone  |    80.776 |    80.624 |    80.281 |    80.624 |   79.768 |   80.29  |
	| CFFL        |    80.856 |    80.861 |    80.49  |    80.803 |   80.17  |   80.544 |

### With gradient cut sum alpha=3
#### Fairness
	|         |   Distriubted |      CFFL |   Contributions_V_final |
	|:--------|--------------:|----------:|------------------------:|
	| P10_0.1 |    -0.0688103 | 0.102803  |              0.0829447  |
	| P10_1.0 |    -0.0780402 | 0.041895  |             -0.0531891  |
	| P20_0.1 |     0.189657  | 0.016183  |             -0.00141581 |
	| P20_1.0 |    -0.232403  | 0.0655583 |              0.010762   |
	| P5_0.1  |     0.081479  | 0.223894  |             -0.222496   |
	| P5_1.0  |     0.101172  | 0.504716  |              0.119565   |
#### Test Accuracy
	|             |   P10_0.1 |   P10_1.0 |   P20_0.1 |   P20_1.0 |   P5_0.1 |   P5_1.0 |
	|:------------|----------:|----------:|----------:|----------:|---------:|---------:|
	| Distributed |  0.815611 |  0.815343 |  0.815879 |  0.816235 | 0.80785  | 0.80959  |
	| Standalone  |  0.808029 |  0.783764 |  0.774487 |  0.799822 | 0.798127 | 0.798974 |
	| CFFL        |  0.815968 |  0.814451 |  0.813872 |  0.814897 | 0.810125 | 0.809857 |


1. [x] - run the full-suite experiment on dataset with nan dropped for Adult - __COMPLETE__

~~See if we can avoid dropping performance at the end of the communication round.~~
- ~~500 perparty (lr0001)~~
- ~~1000* perparty (lr00001)~~ 
- 1000* perparty (lr0001) sum 
- sum AND credit-sum

### 1000* perparty (lr0001) sum
#### Results:
	|         |   Distriubted |     CFFL |   Contributions_V_final |
	|:--------|--------------:|---------:|------------------------:|
	| P10_0.1 |     0.298102  | 0.963566 |                0.676225 |
	| P10_1.0 |     0.0697744 | 0.958764 |                0.664555 |
	| P20_0.1 |    -0.131447  | 0.708341 |                0.619548 |
	| P20_1.0 |    -0.262236  | 0.708472 |                0.591847 |
	| P5_0.1  |     0.165296  | 0.976628 |                0.884048 |
	| P5_1.0  |    -0.0490561 | 0.963449 |                0.823565 |
	|             |   P10_0.1 |   P10_1.0 |   P20_0.1 |   P20_1.0 |   P5_0.1 |   P5_1.0 |
	|:------------|----------:|----------:|----------:|----------:|---------:|---------:|
	| Distributed |    81.204 |    81.61  |    77.627 |    78.466 |   81.195 |   81.418 |
	| Standalone  |    80.571 |    80.566 |    80.08  |    80.049 |   80.553 |   80.245 |
	| CFFL        |    80.954 |    80.999 |    81.088 |    80.999 |   80.915 |   80.642 |

### 1000* perparty (lr0001) credit-sum
#### Results:
	|         |   Distriubted |     CFFL |   Contributions_V_final |
	|:--------|--------------:|---------:|------------------------:|
	| P10_0.1 |   0.370483    | 0.995844 |                0.756615 |
	| P10_1.0 |   0.0286734   | 0.994063 |                0.768469 |
	| P20_0.1 |  -0.000251488 | 0.98523  |                0.820729 |
	| P20_1.0 |   0.326513    | 0.987699 |                0.794247 |
	| P5_0.1  |   0.144283    | 0.995792 |                0.839867 |
	| P5_1.0  |  -0.0844832   | 0.994978 |                0.823163 |
	|             |   P10_0.1 |   P10_1.0 |   P20_0.1 |   P20_1.0 |   P5_0.1 |   P5_1.0 |
	|:------------|----------:|----------:|----------:|----------:|---------:|---------:|
	| Distributed |    81.253 |    81.579 |    75.865 |    77.399 |   81.307 |   81.436 |
	| Standalone  |    80.664 |    80.687 |    80.259 |    80.147 |   80.562 |   80.397 |
	| CFFL        |    80.754 |    80.731 |    80.415 |    80.321 |   80.718 |   80.446 |


### 1000* perparty (lr0001) sum no-pretrain
#### Results:
	|         |   Distriubted |     CFFL |   Contributions_V_final |
	|:--------|--------------:|---------:|------------------------:|
	| P10_0.1 |   0.276184    | 0.951093 |                0.659389 |
	| P10_1.0 |   0.161764    | 0.968217 |                0.684205 |
	| P20_0.1 |  -0.000467303 | 0.639044 |                0.542624 |
	| P20_1.0 |  -0.107185    | 0.659705 |                0.512569 |
	| P5_0.1  |  -0.170295    | 0.959823 |                0.873949 |
	| P5_1.0  |  -0.442646    | 0.928185 |                0.773191 |
	|             |   P10_0.1 |   P10_1.0 |   P20_0.1 |   P20_1.0 |   P5_0.1 |   P5_1.0 |
	|:------------|----------:|----------:|----------:|----------:|---------:|---------:|
	| Distributed |    81.342 |    81.641 |    79.161 |    78.804 |   81.24  |   81.45  |
	| Standalone  |    80.606 |    80.486 |    80.192 |    80.071 |   80.62  |   80.241 |
	| CFFL        |    80.972 |    80.95  |    81.133 |    81.035 |   80.932 |   80.66  |

### 1000* perparty (lr0001) credit-sum no-pretrain
#### Results:
	|         |   Distriubted |     CFFL |   Contributions_V_final |
	|:--------|--------------:|---------:|------------------------:|
	| P10_0.1 |     0.309912  | 0.990409 |                0.787775 |
	| P10_1.0 |    -0.0813792 | 0.996282 |                0.768822 |
	| P20_0.1 |    -0.453333  | 0.983986 |                0.834583 |
	| P20_1.0 |    -0.0555836 | 0.98489  |                0.801629 |
	| P5_0.1  |     0.235268  | 0.994754 |                0.861884 |
	| P5_1.0  |    -0.237416  | 0.992427 |                0.828656 |
	|             |   P10_0.1 |   P10_1.0 |   P20_0.1 |   P20_1.0 |   P5_0.1 |   P5_1.0 |
	|:------------|----------:|----------:|----------:|----------:|---------:|---------:|
	| Distributed |    81.187 |    81.575 |    76.802 |    76.213 |   81.209 |   81.392 |
	| Standalone  |    80.682 |    80.7   |    80.237 |    80.107 |   80.553 |   80.442 |
	| CFFL        |    80.776 |    80.745 |    80.455 |    80.25  |   80.709 |   80.45  |

## Credit Sum  (credit weighted sum)
### Adult
1. [x] -  run credit-sum for the 500 perparty setting to see if it can reach better fairness - __COMPLETE__

#### Results:
	 |         |   Distriubted |     CFFL |   Contributions_V_final |
	 |:--------|--------------:|---------:|------------------------:|
	 | P10_0.1 |    0.0416253  | 0.967811 |                0.842494 |
	 | P10_1.0 |   -0.00243359 | 0.97085  |                0.838971 |
	 | P20_0.1 |   -0.213877   | 0.957852 |                0.829683 |
	 | P20_1.0 |    0.0222218  | 0.960659 |                0.792706 |
	 | P5_0.1  |  nan          | 0.978062 |                0.832128 |
	 | P5_1.0  |    0.0711416  | 0.974996 |                0.78492  |
	 |             |   P10_0.1 |   P10_1.0 |   P20_0.1 |   P20_1.0 |   P5_0.1 |   P5_1.0 |
	 |:------------|----------:|----------:|----------:|----------:|---------:|---------:|
	 | Distributed |    81.343 |    81.378 |    81.784 |    81.857 |   80.423 |   80.937 |
	 | Standalone  |    79.97  |    79.782 |    80.334 |    80.257 |   79.491 |   79.743 |
	 | CFFL        |    80.227 |    80.449 |    80.594 |    80.556 |   79.884 |   80.355 |


2. [x] - run credit-sum for lr=0.001 setting to see if it corrects for the instability - __COMPLETE__

#### Results:
	|         |   Distriubted |     CFFL |   Contributions_V_final |
	|:--------|--------------:|---------:|------------------------:|
	| P10_0.1 |    -0.298843  | 0.9921   |                0.774863 |
	| P10_1.0 |    -0.0718331 | 0.989802 |                0.761198 |
	| P20_0.1 |     0.51305   | 0.988957 |                0.732011 |
	| P20_1.0 |     0.51398   | 0.991257 |                0.719077 |
	| P5_0.1  |    -0.0706579 | 0.986048 |                0.810387 |
	| P5_1.0  |     0.340125  | 0.994423 |                0.791185 |
	|             |   P10_0.1 |   P10_1.0 |   P20_0.1 |   P20_1.0 |   P5_0.1 |   P5_1.0 |
	|:------------|----------:|----------:|----------:|----------:|---------:|---------:|
	| Distributed |    80.826 |    81.091 |    79.773 |    81.292 |   81.292 |   81.262 |
	| Standalone  |    81.288 |    81.176 |    81.51  |    81.425 |   80.971 |   81.168 |
	| CFFL        |    81.514 |    81.425 |    81.608 |    81.532 |   81.074 |   81.253 |

### MNIST imbalanced class numbers setting
1. [x] -  compare with the MNIST credit-sum results to show that it will not be dominated by the least reputable - __COMPLETE__
_Motivation_: though in the upload == 1 cases the performance of the best worker does not suffer, it is highly insecure and not private, so we would want to achieve where the upload is much less and still maintain a high performance

2. [x] - run a case with larger party number on MNIST and credit-sum - __COMPLETE__

- P5 theta =[0.1, 1.0] 
- P10 theta = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0] 

#### Resuts:
	|         |   Distriubted |     CFFL |   Contributions_V_final |
	|:--------|--------------:|---------:|------------------------:|
	| P10_0.1 |   nan         | 0.999999 |                     nan |
	| P10_0.2 |   nan         | 0.999998 |                     nan |
	| P10_0.4 |     0.0814127 | 0.999999 |                     nan |
	| P10_0.6 |    -0.139876  | 0.999999 |                     nan |
	| P10_0.8 |     0.200977  | 0.999999 |                     nan |
	| P10_1.0 |     0.126151  | 0.999999 |                     nan |
	| P5_0.1  |    -0.263989  | 1        |                     nan |
	| P5_1.0  |    -0.185295  | 0.999999 |                     nan |
	|             |   P10_0.1 |   P10_0.2 |   P10_0.4 |   P10_0.6 |   P10_0.8 |   P10_1.0 |   P5_0.1 |   P5_1.0 |
	|:------------|----------:|----------:|----------:|----------:|----------:|----------:|---------:|---------:|
	| Distributed |    10.038 |    10.274 |    10.09  |    10.318 |    10.228 |    10.376 |   55.58  |   70.914 |
	| Standalone  |    85.83  |    85.79  |    85.636 |    85.638 |    85.696 |    85.494 |   86.544 |   86.574 |
	| CFFL        |    85.914 |    85.882 |    85.692 |    85.734 |    85.778 |    85.644 |   86.52  |   86.594 |


3. [x] - Compare MNIST P5, thetas = [0.1, 1] setting between using _direct sum_ VS _credit sum_:

#### Results:
##### Direct sum:
	|        |   Distriubted |       CFFL |   Contributions_V_final |
	|:-------|--------------:|-----------:|------------------------:|
	| P5_0.1 |     0.0336366 | nan        |                     nan |
	| P5_1.0 |   nan         |   0.999385 |                     nan |
	|             |   P5_0.1 |   P5_1.0 |
	|:------------|---------:|---------:|
	| Distributed |   85.59  |   55.278 |
	| Standalone  |   86.716 |   86.738 |
	| CFFL        |   25.414 |   87.964 |

##### Credit sum:
	|        |   Distriubted |     CFFL |   Contributions_V_final |
	|:-------|--------------:|---------:|------------------------:|
	| P5_0.1 |     -0.263989 | 1        |                     nan |
	| P5_1.0 |     -0.185295 | 0.999999 |                     nan |
	|             |   P5_0.1 |   P5_1.0 |
	|:------------|---------:|---------:|
	| Distributed |   55.58  |   70.914 |
	| Standalone  |   86.544 |   86.574 |
	| CFFL        |   86.52  |   86.594 |

_Using credit sum for update aggregation can well mitigate the issue of the most contriubtive party being dominated by another party with the same number of data points, under the setting of theta=0.1. As we can clearly see, under the setting of theta=1, the convergence is what we would expect given different parties have different classes of datapoints since the each party is relatively isolated for its evaluation. But under the setting of theta=0.1, the parties are evaluated together and it causes such learning-hijacking problem._

_However, we note that, in such cases, it seems that the parties are no longer helping each other achieving better performance, we believe it could be caused by the selection of the punishment factor (alpha parameter in the sinh function) and it might need to be decresed in the settings where the contributions of the parties are already very well distinguishable so that a large punishment factor would isolate low-contribution party too early from the collaboration preventing them to learn and helping others learn._


4. [ ] - Run MNIST P5 with smaller alpha to allow low-contribution parties to learn, and yet not hijacking the entire training process. __PENDING__

## No pretrain 
### Adult

1. [x] - conduct no pretrain for 40 communication rounds - __COMPLETE__

#### Results:
	|         |   Distriubted |     CFFL |   Contributions_V_final |
	|:--------|--------------:|---------:|------------------------:|
	| P10_0.1 |      0.128158 | 0.631266 |                0.386762 |
	| P10_1.0 |      0.454375 | 0.456134 |                0.194421 |
	| P20_0.1 |     -0.419014 | 0.627031 |                0.536876 |
	| P20_1.0 |     -0.666683 | 0.772636 |                0.587413 |
	| P5_0.1  |      0.196924 | 0.605719 |                0.55924  |
	| P5_1.0  |      0.453753 | 0.382098 |                0.422874 |
	|             |   P10_0.1 |   P10_1.0 |   P20_0.1 |   P20_1.0 |   P5_0.1 |   P5_1.0 |
	|:------------|----------:|----------:|----------:|----------:|---------:|---------:|
	| Distributed |    81.352 |    81.459 |    79.401 |    78.635 |   80.89  |   80.847 |
	| Standalone  |    80.749 |    80.714 |    80.462 |    80.334 |   80.312 |   80.257 |
	| CFFL        |    81.583 |    81.6   |    81.382 |    81.134 |   80.774 |   80.855 |



2. [x] - conduct no pretrain for full 100 communication rounds - __COMPLETE__

#### Results:
	|         |   Distriubted |     CFFL |   Contributions_V_final |
	|:--------|--------------:|---------:|------------------------:|
	| P10_0.1 |     0.0465178 | 0.88895  |                0.651231 |
	| P10_1.0 |     0.607076  | 0.810996 |                0.628693 |
	| P20_0.1 |    -0.0512543 | 0.817794 |                0.644739 |
	| P20_1.0 |    -0.655356  | 0.851323 |                0.670486 |
	| P5_0.1  |     0.164908  | 0.746662 |                0.711852 |
	| P5_1.0  |     0.414504  | 0.900074 |                0.80051  |
	|             |   P10_0.1 |   P10_1.0 |   P20_0.1 |   P20_1.0 |   P5_0.1 |   P5_1.0 |
	|:------------|----------:|----------:|----------:|----------:|---------:|---------:|
	| Distributed |    81.108 |    81.561 |    80.505 |    75.029 |   80.74  |   80.886 |
	| Standalone  |    80.719 |    80.642 |    80.342 |    80.261 |   80.449 |   80.612 |
	| CFFL        |    81.442 |    81.335 |    81.455 |    81.395 |   80.642 |   80.877 |

## Learning-hijack problem
When multiple parties with data of different qualities collaborate, for example in the case of MNIST in the imbalanced class setting. We observe a phenomenon we are naming the _learning-hijack problem_, where because the low-contribution parties' updates carry the same degree of weight in the server's aggregation logic, the convergence curve even for the high-contribution parties become 'hijacked' and its model converges to low performance. We do note that it is _not_ necessarily the case that this problem cause all models to converge to the standalone models of the low-contribution parties, but rather it simply causes the model performance to deteriorate, due to a large proportion of 'bad' gradient updates being aggregated by the server. This is supported by several observations in which the conditions can lead to such problem:
- With more parties 
- With higher learning rate
- With smaller batch sizes
- With higher local training epochs during the collaborative learning process
- With smaller theta


The first four causes can be all translated to the same effect, that is the magnitude of the aggregated gradient update by the server increases and becomes too large. The last cause of smaller theta is due to the design of our algorithm. In the extreme case of theta=1.0, the server evaluates the parties' contribution individually because it can recover the parties' model entirely (by keeping track of their updates each round and the initial model parameters). As a result, the learning-hijack problem is mitigated. On the other hand, when theta is smalle such as theta=0.1, the server has to aggregate all the updates and evaluate these updates together, and it then leads to the same effect of the aggregated gradient update has too large magnitude.

The observeations are obtained from the experimental results on MNIST dataset with various settings. We include the diagrams and settings here. SEE: mnist/lr01\_various