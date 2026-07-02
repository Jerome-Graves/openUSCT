function uap_gradient_check()
%UAP_GRADIENT_CHECK Validate the MEX core: adjoint gradient vs finite difference.
%   Self-contained (no Python needed). If the column-major <-> row-major layout
%   handling were wrong, this check would fail.

h = 1e-3; n = 41; dt = 1.5e-7; nt = 300; f0 = 0.4e6;

% Ring of 12 elements, 1-based [iy ix] subscripts.
ne = 12; R = 0.016; cx = (n-1)*h/2;
th = linspace(0, 2*pi, ne+1); th(end) = [];
ix = round((cx + R*cos(th))/h) + 1;
iy = round((cx + R*sin(th))/h) + 1;
elem_subs = [iy(:) ix(:)];
tx_sel = 1:ne;

% Ricker wavelet.
t = (0:nt-1)*dt - 1/f0; a = (pi*f0*t).^2; wav = (1 - 2*a).*exp(-a);

% Uniform medium with a small low-velocity flaw.
[Y, X] = ndgrid((0:n-1)*h, (0:n-1)*h);
c = 3000*ones(n, n);
r = hypot(X - 0.60*(n-1)*h, Y - 0.50*(n-1)*h);
c(r <= 0.004) = 2600;
m_true = 1 ./ c.^2;

dobs = uap_forward_fmc(m_true, elem_subs, tx_sel, wav, h, dt, nt);

m0 = 1 ./ (3000*ones(n, n)).^2;
[~, g] = uap_misfit_and_gradient(m0, elem_subs, tx_sel, wav, h, dt, nt, dobs);

probes = [round(0.60*(n-1))+1 round(0.50*(n-1))+1;
          round(n/2)          round(n/2);
          round(n/2)+4         round(n/2)-3];

maxrel = 0;
for p = 1:size(probes,1)
    py = probes(p,1); px = probes(p,2);
    eps = 1e-3 * m0(py,px);
    mp = m0; mp(py,px) = mp(py,px) + eps;
    mm = m0; mm(py,px) = mm(py,px) - eps;
    [Jp, ~] = uap_misfit_and_gradient(mp, elem_subs, tx_sel, wav, h, dt, nt, dobs);
    [Jm, ~] = uap_misfit_and_gradient(mm, elem_subs, tx_sel, wav, h, dt, nt, dobs);
    fd = (Jp - Jm) / (2*eps); ad = g(py,px);
    rel = abs(fd - ad) / max(abs(fd), 1e-30);
    fprintf('probe (%d,%d): fd=%+.6e adj=%+.6e rel=%.3e\n', py, px, fd, ad, rel);
    maxrel = max(maxrel, rel);
end

assert(maxrel < 1e-4, 'MEX gradient check failed: max rel err %.3e', maxrel);
fprintf('MEX gradient check passed (max rel err %.3e)\n', maxrel);
end
