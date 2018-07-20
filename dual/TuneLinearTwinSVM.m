function [ C1_best, C2_best ] = TuneLinearTwinSVM( trainData, trainLabels )
%TUNELINEARTWINSVM 
% This function can be used for tuning parameters of Twin SVM with linear
% kernel. It uses grid search for tuning.
%
% Jayadeva, Khemchandani, R., and Suresh Chandra. 
% "Twin support vector machines for pattern classification." 
% IEEE Transactions on pattern analysis and machine intelligence 29.5 (2007): 905-910.
% Sumit Soman, 20 July 2018
% eez127509@ee.iitd.ac.in


% Initializations
C1_range=[0.001;0.01;0.1;1;10;20;50;100;500];
C2_range=[0.001;0.01;0.1;1;10;20;50;100;500];

% Separate validation set
[N, D]=size(trainData);
split_pt=round(0.8*N);
xTrain=trainData(1:split_pt,:);
yTrain=trainLabels(1:split_pt,:);
xTest=trainData(split_pt+1:end,:);
yTest=trainLabels(split_pt+1:end,:);
bestAcc=0;
C1_best=0;C2_best=0;



% Tune
for i=1:length(C1_range)
    C1=C1_range(i);
    for j=1:length(C2_range)
        C2=C2_range(j);
        
        % Train and test twin SVM
        [ yPred, accuracy, model ] = LinearTWSVM( xTrain, yTrain, xTest, yTest, C1, C2 );
        
        if (accuracy>bestAcc)
            bestAcc=accuracy;
            C1_best=C1;
            C2_best=C2;
        end
    end
end


end

