function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
[a, b] = size(theta);
% You need to return the following variables correctly 
J = 0;
z = X*theta;
h_theta = sigmoid(z);
A = log(h_theta);
B = log(1-h_theta);
regular=0;
for i=2:a
    regular = regular + (lambda*(theta(i,1))^2)/(2*m);
end

J = ((sum(y.*A+ (1-y).*B))/(-1*m)) + regular;
grad = zeros(size(theta));
[c, d] = size(X);
for j = 1:a
    if j == 1
        grad(j,1)= sum((h_theta-y).*X(:,j))/(m);
    else
        grad(j,1)= sum((h_theta-y).*X(:,j))/(m) + (lambda*theta(j,1))/m;  
    end
   

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta





% =============================================================

end
