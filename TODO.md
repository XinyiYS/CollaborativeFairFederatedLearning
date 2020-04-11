- [x] write own worker class to manage non-iid datasets
- [x] experiment: local 5 epochs during each FL epoch, initially locally train 5-10 epochs first
- [x] add in fairness correlation, see pearcor from sklearn
- [x] implement MLP and CNN models follow the paper

- [ ] ~~parallelize the local pretraining, FL-local training and Shapley-fairness calculation~~
- [X] fix the training of MLP and CNN on MNIST to achieve normal test accuracy (96+%)
- [ ] find out why the federated averaging produces model with low val/test acc
- [ ] implement gradient sharing, up/downloading and points