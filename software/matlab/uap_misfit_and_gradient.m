function [J, grad] = uap_misfit_and_gradient(m, elem_subs, tx_sel, wavelet, h, dt, nt, dobs)
%UAP_MISFIT_AND_GRADIENT Waveform misfit and adjoint-state gradient via libuap.
%   Inputs as in uap_forward_fmc, plus dobs (n_tx x nt x n_rec) observed data.
%   Returns the misfit J and the gradient (same shape as m).
dims = size(m);
ndim = numel(dims);
mflat = reshape(permute(m, ndim:-1:1), [], 1);
rec_lin = subs2lin(elem_subs, dims);
tx_lin = rec_lin(tx_sel);
dflat = reshape(permute(dobs, [3 2 1]), [], 1);      % row-major (tx, nt, rec)

[J, gflat] = uap_mex('gradient', mflat, dims(:), h, dt, nt, ...
                     tx_lin, rec_lin, wavelet(:), dflat);

grad = permute(reshape(gflat, fliplr(dims)), ndim:-1:1);
end
