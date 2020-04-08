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

possible_vals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
best_c = 0.01;
best_s = 0.01;
lowest_error = [];

for c = 1:length(possible_vals)
    for s = 1:length(possible_vals)
        C = possible_vals(c);
        sigma = possible_vals(s);
        model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        if isempty(lowest_error)
            lowest_error =+ error;
        end
        if error < lowest_error(1)
            lowest_error = error;
            best_c = C;
            best_s = sigma;
        end
    end
end

C = best_c;
sigma = best_s;





% =========================================================================

end
