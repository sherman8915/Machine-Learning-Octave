function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== Explanation ======================
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. In Part 1is verified by checking that the
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: The backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. It returns the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. Part 2 can be verified by
%         by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X]; %Adding bias unit to input (x0 element)
Z1 = Theta1*X';    % Linear portion of logistic regression for layer 2 units
A2 = sigmoid(Z1); % Calculate logistic regression for layer 2 values
A2 = [ones(1,m); A2]; % Adding bias unit for layer 2 outputs
Z2 = Theta2*A2; % Linear portion of logstic regression for layer 3 units
A3 = sigmoid(Z2); %Calculate logistic regression for layer 3 values

[max_values, predictions] = max(A3); %p is assigned the index with the max value in each column which actually tells what label was predicted
p = predictions';



% A3 - num_labels X m
% y - m X 1
% Y - num_labels X m

%%% generate Y - recoding y to a "num_labels X m" matrix of 0 and 1 
Y = zeros(num_labels,m);
for i=1:m
	j = y(i, 1);
	Y(j, i) = 1;
endfor
%%% generate Y

%%% Initialize theta_0 regularization parameters
Theta1_reg = Theta1;
Theta2_reg = Theta2;
Theta1_reg(:,1) = 0;
Theta2_reg(:,1) = 0;
theta_reg = [Theta1_reg(:); Theta2_reg(:)];
%%% Initialize theta_0

H = A3; % the hyphothesis g(theta'*x) or g(X* theta) in neural network case that would be A_l
diff = (-Y .* log(H)) - ((1 - Y) .* log(1 - H));
J_original = (sum(sum(diff)))/m ;
J_reg = sum((lambda/(2*m))*(theta_reg.^2));
J = J_original + J_reg';

%%%% Backpropagation calculation
D_1 = Theta1_grad;
D_2 = Theta2_grad;
for t=1:m
	a1 = X'(:, t);
	z1 = Z1(:, t);
	a2 = A2(:, t);
	z2 = Z2(:, t);
	a3 = A3(:, t);
	y = Y(:,t);
	delta_3 = a3 - y;
	delta_2 = (Theta2' * delta_3) .*  (a2 .* (1 - a2));
	delta_2 = delta_2(2:end); % omitting delta for bias term
	D_1 = D_1 + delta_2 * a1';
	D_2 = D_2 + delta_3 * a2';	
endfor 
Theta1_grad = (D_1 / m) + (lambda * Theta1_reg / m);
Theta2_grad = (D_2 / m) + (lambda * Theta2_reg / m);
%%%% Backpropagation









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
