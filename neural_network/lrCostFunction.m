function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== Explanation ======================
% Description: Computes the cost of a particular choice of theta and its gradient.
%               Returns: 
%			J - the cost.
%			grad - gradient calculated from partial derivatives
%
% Additional Info:
% 	Cost: 
%		The computation of the cost function and gradients can be
%       	efficiently vectorized. For example, consider the computation
%
%           	sigmoid(X * theta)
%
%       	Each row of the resulting matrix will contain the value of the
%       	prediction for that example. You can make use of this to vectorize
%      		the cost function and gradient computations. 
%
% 	Regularization: 
%		When computing the gradient of the regularized cost function, 
%       	there're many possible vectorized solutions, but one solution
%       	looks like:
%           		grad = (unregularized gradient for logistic regression)
%           		temp = theta; 
%           		temp(1) = 0;   % because we don't add anything for j = 0  
%           		grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% Note:
% size(X) -   [m , n+1]     
% size(y) -   [m , 1]
% size(Z) -   [m , 1]
% size(H) -   [m , 1]

m = length(y);
n_1 = size(X)(1,2);
Z = X*theta;  % The Linear portion of the hyphothesis result vector
H = sigmoid(Z); % the hyphothes is g(theta'*x) or g(X* theta)
diff = (-y .* log(H)) - ((1 - y) .* log(1 - H));
J_original = (sum(diff))/m ;
theta_reg = theta;
theta_reg(1,1) = 0;
J_reg = sum((lambda/(2*m))*(theta_reg.^2));
J = J_original + J_reg';

D = (H - y) .* X;
grad_orig = sum(D)/m;
grad_reg = (lambda/m) * theta_reg';
grad = grad_orig + grad_reg;








% =============================================================

grad = grad(:);

end
