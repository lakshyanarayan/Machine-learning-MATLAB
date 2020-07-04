function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%   LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%   regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
[a, b] = size(theta);
regular=0;
for i=2:a
    regular = regular + (lambda*(theta(i,1))^2)/(2*m);
end

h=X*theta;
J=(1/(2*m))*sum((h-y).^2) + regular;
for j = 1:a
    if j == 1
        grad(j,1)= sum((h-y).*X(:,j))/(m);
    else
        grad(j,1)= sum((h-y).*X(:,j))/(m) + (lambda*theta(j,1))/m;  
    end
   

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
