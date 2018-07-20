function [ yPred, accuracy ] = KernelTWSVM( xTrain, yTrain, xTest, yTest, C1, C2, kernel_type, kernel_param )
%KERNELTWSVM - This function implements the kernel Twin SVM formulation (dual) for binary classification.
% Inputs: 
% xTrain: Training data (samplesXfeatures)
% yTrain: Training labels (samplesX1) - should be +1/-1
% xTest: Testing data (test_samplesXfeatures)
% yTest: Testing labels (test_samplesX1) - should be +1/-1
% C1, C2: Hyperparameters for the two hyperplanes
% kernel_type: Kernel Type 1: Linear, 2: Polynomial, 3:RBF
% kernel_param: Kernel Parameters: degree for kernel_type=2, RBF width for
%               kernel_type=3
%
% Jayadeva, Khemchandani, R., and Suresh Chandra. 
% "Twin support vector machines for pattern classification." 
% IEEE Transactions on pattern analysis and machine intelligence 29.5 (2007): 905-910.
% Sumit Soman, 20 July 2018
% eez127509@ee.iitd.ac.in


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
N1=size(A,1);
N2=size(B,1);


% Compute training and test kernels
for i=1:size(A,1)
    for j=1:size(xTrain,1)
        KA(i,j)=kernelfunction(kernel_type,A(i,:),xTrain(j,:),kernel_param);
    end
end

for i=1:size(B,1)
    for j=1:size(xTrain,1)
        KB(i,j)=kernelfunction(kernel_type,B(i,:),xTrain(j,:),kernel_param);
    end
end

for i=1:size(xTest,1)
    for j=1:size(xTrain,1)
        Ktest(i,j)=kernelfunction(kernel_type,xTest(i,:),xTrain(j,:),kernel_param);
    end
end

% Obtain Twin SVM hyperplanes
[ wA, bA, EXITFLAG1 ] = LTWSVM1( KA, KB, C1 );
[ wB, bB, EXITFLAG2 ] = LTWSVM2( KA, KB, C2 );

if (EXITFLAG1~=1 || EXITFLAG2~=1)
    fprintf(1, 'Optimization did not converge! --- EXITFLAG1 = %d --- EXITFLAG2 = %d\n', EXITFLAG1, EXITFLAG2);
    wA=rand(N1+N2,1);bA=rand;
    wB=rand(N1+N2,1);bB=rand;
end

% Compute test set predictions
yPred=zeros(size(xTest,1),1);
for i=1:size(xTest,1)
    sample=Ktest(i,:);
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

