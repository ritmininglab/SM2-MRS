function g = softmax(z)
%Softmax input shape = nbatch*ndim
g = exp(z) ./ repmat(sum(exp(z),2), 1, size(z,2));
end