This folder contains the code for MATLAB implementation of the dual formulation of the Twin Support Vector Machine.

Reference Paper:
Jayadeva, Khemchandani, R., and Suresh Chandra. "<a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4135685">Twin support vector machines for pattern classification.</a>" IEEE Transactions on pattern analysis and machine intelligence 29.5 (2007): 905-910.

Specifically, equations (28) and (29) from the paper are implemented in this code. Please see `sample_script.m` for a sample implementation of the Twin SVM with linear and RBF kernels on the sample dataset in `data_heartstatlog.m`

The following are the files available:

1. `LTWSVM1.m` and `LTWSVM2.m` - Implementation of equations (28) and (29) from the paper.
2. `LinearTWSVM.m` - This function implements the linear Twin SVM formulation (dual) for binary classification.
3. `KernelTWSVM.m` - This function implements the kernel Twin SVM formulation (dual) for binary classification.
4. `TuneLinearTwinSVM.m` - This function can be used for tuning parameters of Twin SVM with linear kernel. It uses grid search for tuning.
5. `kernelfunction.m` - This function computes linear, polynomial or Radial Basis Function (RBF) kernel.
6. `data_heartstatlog.m` - Sample dataset
7. `sample_script.m` = Sample code for running Twin SVM with linear and RBF kernels on heart statlog dataset


Note: For tuning the parameters, one can use the function `TuneLinearTwinSVM.m`, where tuning of parameters for a linear kernel Twin SVM is shown. Here, the parameters `C_1` and `C_2` are tuned by grid search and the parameters which give the best accuracy on the validation set (chosen as 20% of the training set in this function) are chosen as the optimal parameters for training the final model. In case of using any other kernel, the kernel parameter (degree for polynomial kernel and kernel width for RBF kernel) can also be tuned by suitably editing this function.
