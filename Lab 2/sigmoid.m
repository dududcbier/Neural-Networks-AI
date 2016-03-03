%this functions calculates the sigmoid
function [output] = sigmoid(x)
    output = 1 ./ (1 + exp(-x));
end

