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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1a: Calculating cost using formula
for i = 1:m
	one_training_eg = X(i,:);
	a1 = one_training_eg'; % Input
	a1 = [1; a1]; % Bias term

	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	a2 = [1; a2]; % Bias term

	z3 = Theta2 * a2;
	a3 = sigmoid(z3); % Output

	mod_y = zeros(num_labels, 1); 
	mod_y(y(i)) = 1; % Modified y so that it is a vector
	cost = sum(mod_y .* log(a3) + (1 - mod_y) .* log(1 - a3));
	J += cost;
end
J = J / (-m);

% Part 1b: Calculating regularized cost function
mod_theta1 = Theta1(:,2:end); % Remove bias term
mod_theta2 = Theta2(:,2:end); % Remove bias term

cost = sum(sumsq(mod_theta1)) + (sum(sumsq(mod_theta2)));
cost *= (lambda / (2 * m));
J += cost;


% Part 2: Backpropagation
for t = 1:m
	one_training_eg = X(t,:);
	a1 = one_training_eg'; % Input
	a1 = [1; a1]; % Bias term

	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	a2 = [1; a2]; % Bias term

	z3 = Theta2 * a2;
	a3 = sigmoid(z3); % Output

	mod_y = zeros(num_labels, 1); 
	mod_y(y(t)) = 1; % Modified y so that it is a vector
	% Alternative: Logical arrays y = (y == y(t))

	% Step 1 to Step 5 in ex4
	delta_3 = a3 - mod_y;
	delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z2]);
	delta_2 = delta_2(2:end);

	Theta2_grad = Theta2_grad + delta_3 * a2';
	Theta1_grad = Theta1_grad + delta_2 * a1';
end

Theta2_grad /= m;
Theta1_grad /= m;


% Part 3: Regularized neural network
Theta1(:,1) = 0;
Theta1 *= (lambda / m);
Theta1_grad += Theta1;

Theta2(:,1) = 0;
Theta2 *= (lambda / m);
Theta2_grad += Theta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
