// pybind11 bindings for the libuap C++ core.
//
// Exposes the acoustic forward model and the exact adjoint-state FWI gradient
// to Python as functions over NumPy arrays. Grid indices arrive as flat linear
// indices (row-major), which the Python wrapper computes from geometry.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

#include "libuap.hpp"

namespace py = pybind11;

using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IArr = py::array_t<int, py::array::c_style | py::array::forcecast>;

static std::vector<int> shape_of(const Arr& m) {
    std::vector<int> dims;
    for (py::ssize_t d = 0; d < m.ndim(); ++d) dims.push_back(static_cast<int>(m.shape(d)));
    return dims;
}

// forward_fmc(m, h, dt, nt, tx_lin, rec_lin, wavelet) -> data[n_tx, nt, n_rec]
static Arr forward_fmc(Arr m, double h, double dt, int nt,
                       IArr tx_lin, IArr rec_lin, Arr wavelet) {
    auto dims = shape_of(m);
    int ndim = static_cast<int>(dims.size());
    int n_tx = static_cast<int>(tx_lin.size());
    int n_rec = static_cast<int>(rec_lin.size());

    Arr data({n_tx, nt, n_rec});
    uap::forward_fmc(m.data(), dims.data(), ndim, h, dt, nt,
                     tx_lin.data(), n_tx, rec_lin.data(), n_rec,
                     wavelet.data(), data.mutable_data());
    return data;
}

// misfit_and_gradient(...) -> (J, grad[shape of m])
// misfit_type: 0 = least squares, 1 = global correlation norm (GCN).
static py::tuple misfit_and_gradient(Arr m, double h, double dt, int nt,
                                     IArr tx_lin, IArr rec_lin, Arr wavelet, Arr dobs,
                                     int misfit_type) {
    auto dims = shape_of(m);
    int ndim = static_cast<int>(dims.size());
    int n_tx = static_cast<int>(tx_lin.size());
    int n_rec = static_cast<int>(rec_lin.size());

    Arr grad(dims);
    double J = uap::misfit_and_gradient(m.data(), dims.data(), ndim, h, dt, nt,
                                        tx_lin.data(), n_tx, rec_lin.data(), n_rec,
                                        wavelet.data(), dobs.data(), grad.mutable_data(),
                                        misfit_type);
    return py::make_tuple(J, grad);
}

PYBIND11_MODULE(_uap, mod) {
    mod.doc() = "libuap: C++ acoustic forward model and adjoint-state FWI gradient";
    mod.def("forward_fmc", &forward_fmc,
            py::arg("m"), py::arg("h"), py::arg("dt"), py::arg("nt"),
            py::arg("tx_lin"), py::arg("rec_lin"), py::arg("wavelet"));
    mod.def("misfit_and_gradient", &misfit_and_gradient,
            py::arg("m"), py::arg("h"), py::arg("dt"), py::arg("nt"),
            py::arg("tx_lin"), py::arg("rec_lin"), py::arg("wavelet"), py::arg("dobs"),
            py::arg("misfit_type") = 0);
}
