% Sample script for running Twin Support Vector Machine
%
% Jayadeva, Khemchandani, R., and Suresh Chandra. 
% "Twin support vector machines for pattern classification." 
% IEEE Transactions on pattern analysis and machine intelligence 29.5 (2007): 905-910.
% Sumit Soman, 20 July 2018
% eez127509@ee.iitd.ac.in


clc;clearvars;close all;
rng default;

% Read Data
run('data_heartstatlog.m');

% Get Data and Labels
features=data(:,1:end-1);
labels=data(:,end);

% Normalize labels
labels(labels==2)=-1;


% Separate training and test data (80:20 split)
total_samples=size(features,1);
train_samples=round(0.8*total_samples);

% Define training and test samples
xTrain=features(1:train_samples,:);
yTrain=labels(1:train_samples,:);
xTest=features(train_samples+1:end,:);
yTest=labels(train_samples+1:end,:);

% Define hyperparameter values
C1=0.1; C2=0.05;

% Run Twin SVM (Linear)
 [ yPred, accuracy ] = LinearTWSVM( xTrain, yTrain, xTest, yTest, C1, C2 );
 
 disp('Accuracy (Linear) is');
 accuracy
 
 
% Run Twin SVM (Kernel)
kernel_type=3; kernel_param=0.95;

% Run Kernel Twin SVM
[ yPred, accuracy ] = KernelTWSVM( xTrain, yTrain, xTest, yTest, C1, C2, kernel_type, kernel_param );
 
disp('Accuracy (Kernel) is');
accuracy
 
 
