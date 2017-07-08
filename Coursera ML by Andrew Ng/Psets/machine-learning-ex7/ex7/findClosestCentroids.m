function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = size(X, 1);

% Original faulty implementation that resulted in recovered data of only 2/3 colors
#{
idxx = zeros(size(X,1), 1);
lowest_cost = intmax() * ones(m, 1);
for i = 1:m
	for j = 1:K
		x_i = X(i,:);
		centroid_j = centroids(j,:);
		cost = sumsq(x_i - centroid_j); % Actual formula need to multiply (1 / m)

		if cost < lowest_cost(i)
			lowest_cost(i) = cost;
			idxx(i) = j;
		end
	end
end
#}
% Solution given by benoitvallon
% (https://github.com/benoitvallon/coursera-machine-learning)
for i = 1:m
  	x = X(i,:);
  	norms = zeros(K, 1);
  	for centroid_i = 1:K
    	norms(centroid_i) = (x - centroids(centroid_i,:)) * (x - centroids(centroid_i,:))';
  	end
  	[value, idx(i)] = min(norms);
end

% =============================================================

end

