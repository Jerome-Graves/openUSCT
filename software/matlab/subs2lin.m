function lin = subs2lin(subs, dims)
%SUBS2LIN 0-based row-major linear indices from 1-based subscripts.
%   subs : n x ndim matrix of 1-based subscripts, columns ordered as dims
%   dims : 1 x ndim size vector (as returned by size())
ndim = numel(dims);
strides = zeros(1, ndim);
strides(ndim) = 1;
for a = ndim-1:-1:1
    strides(a) = strides(a+1) * dims(a+1);
end
lin = sum((subs - 1) .* strides, 2);
end
