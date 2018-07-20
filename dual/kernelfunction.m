function[k]=kernelfunction(type,u,v,p)

% Function to compute kernel
% This function computes linear, polynomial and RBF kernels
% Inputs:
% type: Kernel type (1:linear, 2:polynomial, 3:RBF
% u: Sample x_i for computing kernel
% v: Sample x_j for computing kernel
% p: Kernel parameter (degree for polynomial, kernel width for RBF

% Sumit Soman, 20 July 2018
% eez127509@ee.iitd.ac.in

switch type
    
    case 1;
        %Linear Kernal
        k = u*v';
        
    case 2;
        %Polynomial Kernal
        k = (u*v' + 1)^p;
        
    case 3;
        %Radial Basia Function Kernal
        k = exp(-(u-v)*(u-v)'/(p.^2));
        
    otherwise
        k=0;
        
end
return
