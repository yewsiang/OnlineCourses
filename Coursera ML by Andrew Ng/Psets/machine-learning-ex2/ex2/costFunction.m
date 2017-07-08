function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Apply the sigmoid function on X * theta
h_theta = sigmoid(X * theta);
#{
% Find the rows which have y = 1 and y = 0 to apply different algos
% For y = 1: cost for that entry = log(h(theta))
% For y = 0: cost for that entry = log(1 - h(theta))
pos = find(y == 0);
neg = find(y == 1);

for i = 1:size(pos, 1)
	% Update J(theta) accordingly with each entry
	J = J - log(h_theta(pos(i)));
end

for i = 1:size(neg, 1)
	% Update J(theta) accordingly with each entry
	J = J - log(1 - h_theta(neg(i)));
end
#}
J = sum(- (y .* log(h_theta)) - ((1 - y) .* log(1 - h_theta)))

% Divide by m (According to formula)
J = J / m;


% Change the gradient
grad = sum((h_theta - y) .* X, 1);
grad = grad / m;

% =============================================================

end
