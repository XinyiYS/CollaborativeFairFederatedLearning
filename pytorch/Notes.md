
## Credit Sum  (credit weighted sum)
### Adult
1. [x] -  run credit-sum for the 500 perparty setting to see if it can reach better fairness - COMPLETE

	Results:
	          Distriubted      CFFL  Contributions_V_final
	                                                      
	 P10_0.1     0.041625  0.967811               0.842494
	 P10_1.0    -0.002434  0.970850               0.838971
	 P20_0.1    -0.213877  0.957852               0.829683
	 P20_1.0     0.022222  0.960659               0.792706
	 P5_0.1           NaN  0.978062               0.832128
	 P5_1.0      0.071142  0.974996               0.784920
	              P10_0.1  P10_1.0  P20_0.1  P20_1.0  P5_0.1  P5_1.0
	 Distributed    81.34    81.38    81.78    81.86   80.42   80.94
	 Standalone     79.97    79.78    80.33    80.26   79.49   79.74
	 CFFL           80.23    80.45    80.59    80.56   79.88   80.36
 

2. [x] - run credit-sum for lr=0.001 setting to see if it corrects for the instability - COMPLETE

	Results:
	         Distriubted      CFFL  Contributions_V_final
	                                                     
	P10_0.1    -0.298843  0.992100               0.774863
	P10_1.0    -0.071833  0.989802               0.761198
	P20_0.1     0.513050  0.988957               0.732011
	P20_1.0     0.513980  0.991257               0.719077
	P5_0.1     -0.070658  0.986048               0.810387
	P5_1.0      0.340125  0.994423               0.791185
	             P10_0.1  P10_1.0  P20_0.1  P20_1.0  P5_0.1  P5_1.0
	Distributed    80.83    81.09    79.77    81.29   81.29   81.26
	Standalone     81.29    81.18    81.51    81.42   80.97   81.17
	CFFL           81.51    81.42    81.61    81.53   81.07   81.25

### MNIST
1. [ ] -  compare with the MNIST credit-sum results to show that it will not be dominated by the least reputable - RUNNING
_Motivation_: though in the upload == 1 cases the performance of the best worker does not suffer, it is highly insecure and not private, so we would want to achieve where the upload is much less and still maintain a high performance

2. [ ] -  run a case with larger party number on MNIST and credit-sum - RUNNING
2.1 P10 theta = [0.1, 0.2, 0.4]
2.2 P10 theta = [0.6, 0.8, 1.0]

## No pretrain 
### Adult

1. [x] - conduct no pretrain for 40 communication rounds - COMPLETE

	Results:
	         Distriubted      CFFL  Contributions_V_final
	                                                     
	P10_0.1     0.128158  0.631266               0.386762
	P10_1.0     0.454375  0.456134               0.194421
	P20_0.1    -0.419014  0.627031               0.536876
	P20_1.0    -0.666683  0.772636               0.587413
	P5_0.1      0.196924  0.605719               0.559240
	P5_1.0      0.453753  0.382098               0.422874
	             P10_0.1  P10_1.0  P20_0.1  P20_1.0  P5_0.1  P5_1.0
	Distributed    81.35    81.46    79.40    78.64   80.89   80.85
	Standalone     80.75    80.71    80.46    80.33   80.31   80.26
	CFFL           81.58    81.60    81.38    81.13   80.77   80.86

2. [x] - conduct no pretrain for full 100 communication rounds - COMPLETE

	Results:
	         Distriubted      CFFL  Contributions_V_final
	                                                     
	P10_0.1     0.046518  0.888950               0.651231
	P10_1.0     0.607076  0.810996               0.628693
	P20_0.1    -0.051254  0.817794               0.644739
	P20_1.0    -0.655356  0.851323               0.670486
	P5_0.1      0.164908  0.746662               0.711852
	P5_1.0      0.414504  0.900074               0.800510
	             P10_0.1  P10_1.0  P20_0.1  P20_1.0  P5_0.1  P5_1.0
	Distributed    81.11    81.56    80.50    75.03   80.74   80.89
	Standalone     80.72    80.64    80.34    80.26   80.45   80.61
	CFFL           81.44    81.33    81.45    81.39   80.64   80.88
