function data = uap_forward_fmc(m, elem_subs, tx_sel, wavelet, h, dt, nt)
%UAP_FORWARD_FMC Full-matrix-capture forward modelling via the libuap MEX core.
%   m         : squared-slowness field (2D or 3D array), m = 1/c^2
%   elem_subs : n_elem x ndim matrix of 1-based grid subscripts for elements
%   tx_sel    : indices (1-based) into elem_subs of the transmitting elements
%   wavelet   : transmit wavelet (length nt)
%   h, dt, nt : grid spacing, time step, number of samples
%   data      : n_tx x nt x n_rec array
dims = size(m);
ndim = numel(dims);
mflat = reshape(permute(m, ndim:-1:1), [], 1);      % column-major -> row-major
rec_lin = subs2lin(elem_subs, dims);
tx_lin = rec_lin(tx_sel);

flat = uap_mex('forward', mflat, dims(:), h, dt, nt, tx_lin, rec_lin, wavelet(:));

n_tx = numel(tx_sel);
n_rec = size(elem_subs, 1);
data = permute(reshape(flat, [n_rec nt n_tx]), [3 2 1]);
end
