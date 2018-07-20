function [ yPred, accuracy, model ] = LinearTWSVM( xTrain, yTrain, xTest, yTest, C1, C2 )
%LINEARTWSVM
% This function implements the linear Twin SVM formulation (dual) for binary classification.
% Inputs:
% xTrain: Training data (samplesXfeatures)
% yTrain: Training labels (samplesX1) - should be +1/-1
% xTest: Testing data (test_samplesXfeatures)
% yTest: Testing labels (test_samplesX1) - should be +1/-1
% C1, C2: Hyperparameters for the two hyperplanes
%
% Jayadeva, Khemchandani, R., and Suresh Chandra. 
% "Twin support vector machines for pattern classification." 
% IEEE Transactions on pattern analysis and machine intelligence 29.5 (2007): 905-910.
% Sumit Soman, 20 July 2018
% eez127509@ee.iitd.ac.in

[N,D]=size(xTrain);

% Pre-process data to make zero mean and unit variance
trainmean=mean(xTrain);
trainvar=var(xTrain);
for i=1:size(xTrain,1)
    xTrain(i,:)=(xTrain(i,:)-trainmean)./trainvar; %Normalize train data
end
for i=1:size(xTest,1)
    xTest(i,:)=(xTest(i,:)-trainmean)./trainvar; %Normalize test data
end


% Separate data of the two classes
A=xTrain(yTrain==1,:);
B=xTrain(yTrain==-1,:);

% Obtain Twin SVM hyperplanes
[ wA, bA, EXITFLAG1 ] = LTWSVM1( A, B, C1 );
[ wB, bB, EXITFLAG2 ] = LTWSVM2( A, B, C2 );

if (EXITFLAG1~=1 || EXITFLAG2~=1)
    fprintf(1, 'Optimization did not converge! --- EXITFLAG1 = %d --- EXITFLAG2 = %d', EXITFLAG1, EXITFLAG2);
end
if (all(wA)==0)
    wA=rand(D,1);bA=rand;
end
if (all(wB)==0)
    wB=rand(D,1); bB=rand;
end


model.wA=wA;
model.wB=wB;
model.bA=bA;
model.bB=bB;
model.trainMean=trainmean;
model.trainVar=trainvar;

% Compute test set predictions
yPred=zeros(size(xTest,1),1);
for i=1:size(xTest,1)
    sample=xTest(i,:);
    distA=(sample*wA + bA)/norm(wA);
    distB=(sample*wB + bB)/norm(wB);
    if (distA>distB)
        yPred(i)=-1;
    else
        yPred(i)=1;
    end
end



accuracy=(sum(yPred==yTest)/length(yTest))*100;

% Sanity check - if labels are predicted wrongly then flip
if (accuracy<50)
    yPred=-1*yPred;
    accuracy=(sum(yPred==yTest)/length(yTest))*100;
end
end

