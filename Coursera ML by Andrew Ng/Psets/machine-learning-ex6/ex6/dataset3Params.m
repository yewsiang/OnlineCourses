function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
#{
values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
errors = zeros(length(values), length(values));

for i = 1:length(values)
	for j = 1:length(values)
		% Get the model for each combination of C and sigma
		curr_C = values(i);
		curr_sigma = values(j);
		model = svmTrain(X, y, curr_C, @(x1, x2) gaussianKernel(x1, x2, curr_sigma));

		% Calculate the error of their predictions
		predictions = svmPredict(model, Xval);
		pred_error = mean(double(predictions ~= yval)); % Error whenever prediction != yval
		errors(i, j) = pred_error;
	end
end

% Choose the combination of C & sigma value that results in lowest error
[row, col] = find(errors == min(min(errors))); % Position of min in 1D terms

% There may be several combinations of C & sigma that result in equal mins
% Choose the first available combination
if length(row) > 1
	row = row(1);
end
if length(col) > 1
	col = col(1);
end

C = values(row);
sigma = values(col);
#}
% =========================================================================

end
