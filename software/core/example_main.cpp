// Minimal native C++ example for libuap: no Python, no MATLAB.
//
// Builds a small 2D model with a flaw, runs a full-matrix-capture forward, then
// one misfit-and-gradient evaluation, and prints checksums. Demonstrates that
// the core is usable directly from C++.
//
// Build with CMake (see CMakeLists.txt), or directly:
//   cl /O2 /EHsc /std:c++14 example_main.cpp     (MSVC)
//   g++ -O2 -std=c++14 example_main.cpp -o example_main   (GCC/Clang)

#include <cmath>
#include <cstdio>
#include <vector>

#include "libuap.hpp"

constexpr double PI = 3.14159265358979323846;

int main() {
    const int n = 41;
    const int dims[2] = {n, n};
    const double h = 1.0e-3, dt = 1.5e-7;
    const int nt = 300;
    const double f0 = 0.4e6;
    const std::size_t N = uap::cell_count(dims, 2);

    // Ricker wavelet.
    std::vector<double> wavelet(nt);
    for (int i = 0; i < nt; ++i) {
        double t = i * dt - 1.0 / f0;
        double a = (PI * f0 * t) * (PI * f0 * t);
        wavelet[i] = (1.0 - 2.0 * a) * std::exp(-a);
    }

    // Uniform 3000 m/s medium with a low-velocity circular flaw.
    std::vector<double> m(N);
    const double cx = (n - 1) * h / 2.0;
    for (int iy = 0; iy < n; ++iy)
        for (int ix = 0; ix < n; ++ix) {
            double x = ix * h, y = iy * h;
            double r = std::hypot(x - 0.60 * (n - 1) * h, y - 0.50 * (n - 1) * h);
            double c = (r <= 0.004) ? 2600.0 : 3000.0;
            m[static_cast<std::size_t>(iy) * n + ix] = 1.0 / (c * c);
        }

    // A ring of 12 elements (linear indices).
    const int ne = 12;
    std::vector<int> elem(ne);
    for (int k = 0; k < ne; ++k) {
        double th = 2.0 * PI * k / ne;
        int ix = static_cast<int>(std::lround((cx + 0.016 * std::cos(th)) / h));
        int iy = static_cast<int>(std::lround((cx + 0.016 * std::sin(th)) / h));
        elem[k] = iy * n + ix;
    }

    // Forward.
    std::vector<double> data(static_cast<std::size_t>(ne) * nt * ne);
    uap::forward_fmc(m.data(), dims, 2, h, dt, nt, elem.data(), ne, elem.data(), ne,
                     wavelet.data(), data.data());
    double s = 0.0;
    for (double v : data) s += std::abs(v);
    std::printf("forward: sum|data| = %.6e\n", s);

    // Misfit and gradient from a uniform starting model, with the flaw data as target.
    std::vector<double> m0(N, 1.0 / (3000.0 * 3000.0)), grad(N);
    double J = uap::misfit_and_gradient(m0.data(), dims, 2, h, dt, nt, elem.data(), ne,
                                        elem.data(), ne, wavelet.data(), data.data(), grad.data());
    double gs = 0.0;
    for (double v : grad) gs += std::abs(v);
    std::printf("misfit J = %.6e, sum|grad| = %.6e\n", J, gs);
    std::printf("libuap native example OK\n");
    return 0;
}
