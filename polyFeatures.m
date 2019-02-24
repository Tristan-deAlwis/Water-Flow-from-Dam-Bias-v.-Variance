function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%

X_poly = zeros(numel(X), p);
m = size(X, 1);

for i=1:m
    poly_feature = zeros(p, 1);

    for j=1:p
        poly_feature(j) =  X(i).^j;
    end

    X_poly(i, :) = poly_feature;
end

end

