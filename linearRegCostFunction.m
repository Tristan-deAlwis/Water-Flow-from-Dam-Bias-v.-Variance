function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

%Computing (unregularized) cost function
hyp = (X * theta);
errors = hyp - y;
grad = (X'*(errors))/m; % 2 x 1 matrix
J = sum((errors) .^ 2 / (2 * m));

%Computing cost regularization and adding it to cost function
theta(1) = 0;
J = J + sum(theta.^2) * (lambda / (2 * m));

%Gradient Regularization and adding it to grad
grad = grad + theta * (lambda / m);

grad = grad(:);

end
