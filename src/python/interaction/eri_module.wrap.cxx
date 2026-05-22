
// C.f. https://numpy.org/doc/1.21/reference/c-api/array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL _cpp2py_ARRAY_API
#ifndef CLAIR_C2PY_WRAP_GEN
#ifdef __clang__
// #pragma clang diagnostic ignored "-W#warnings"
#endif
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma GCC diagnostic ignored "-Wcpp"
#endif

#define C2PY_VERSION_MAJOR 0
#define C2PY_VERSION_MINOR 1

#include <c2py/c2py.hpp>

using c2py::operator""_a;

// ==================== Wrapped classes =====================

template <> constexpr bool c2py::is_wrapped<coqui_py::ThcCoulomb> = true;
template <> constexpr bool c2py::is_wrapped<coqui_py::CholCoulomb> = true;

// ==================== enums =====================

// ==================== module classes =====================

template <>
inline constexpr auto c2py::tp_name<coqui_py::ThcCoulomb> =
    "eri_module.ThcCoulomb";
static auto init_0 = c2py::dispatcher_c_kw_t{
    c2py::c_constructor<coqui_py::ThcCoulomb, const coqui_py::Mf &,
                        const std::string &>("mf", "thc_params")};
template <>
constexpr initproc c2py::tp_init<coqui_py::ThcCoulomb> =
    c2py::pyfkw_constructor<init_0>;
template <>
const std::string c2py::tp_ctor_doc<coqui_py::ThcCoulomb> =
    init_0.doc(R"DOC()DOC");
// Np
static auto const fun_0 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) { return self.Np(); }, "self")};

// init
static auto const fun_1 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb &self) { return self.init(); }, "self")};

// initialized
static auto const fun_2 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) { return self.initialized(); },
    "self")};

// mf
static auto const fun_3 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) { return self.mf(); }, "self")};

// mpi
static auto const fun_4 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) { return self.mpi(); }, "self")};

// nbnd
static auto const fun_5 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) { return self.nbnd(); }, "self")};

// nkpts
static auto const fun_6 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) { return self.nkpts(); }, "self")};

// nkpts_ibz
static auto const fun_7 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) { return self.nkpts_ibz(); }, "self")};

// nqpts
static auto const fun_8 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) { return self.nqpts(); }, "self")};

// nqpts_ibz
static auto const fun_9 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) { return self.nqpts_ibz(); }, "self")};

// nspin
static auto const fun_10 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) { return self.nspin(); }, "self")};

// nspin_in_basis
static auto const fun_11 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) { return self.nspin_in_basis(); },
    "self")};

// compute_delta_Xi
static auto const fun_thc_DeltaXi = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb &self,
       const std::string &Deltapsi_prefix,
       nda::array<double, 1> const &q_vec_cryst,
       nda::array<double, 2> const &delta_r_P,
       std::optional<nda::array<double, 1>> const &q_phonon_cryst) {
      return self.compute_delta_Xi(Deltapsi_prefix, q_vec_cryst, delta_r_P, q_phonon_cryst);
    },
    "self", "Deltapsi_prefix", "q_vec_cryst", "delta_r_P", "q_phonon_cryst")};

// compute_delta_V
static auto const fun_thc_dV = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb &self,
       const std::string &Deltapsi_prefix,
       nda::array<double, 1> const &q_vec_cryst,
       nda::array<double, 2> const &delta_r_P,
       std::optional<nda::array<double, 1>> const &q_phonon_cryst) {
      return self.compute_delta_V(Deltapsi_prefix, q_vec_cryst, delta_r_P, q_phonon_cryst);
    },
    "self", "Deltapsi_prefix", "q_vec_cryst", "delta_r_P", "q_phonon_cryst")};

// compute_delta_V_batched
static auto const fun_thc_dV_batched = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb &self,
       const std::string &Deltapsi_prefix,
       nda::array<double, 2> const &q_vec_cryst_list,
       nda::array<double, 2> const &delta_r_P,
       std::optional<nda::array<double, 1>> const &q_phonon_cryst) {
      return self.compute_delta_V_batched(Deltapsi_prefix, q_vec_cryst_list, delta_r_P, q_phonon_cryst);
    },
    "self", "Deltapsi_prefix", "q_vec_cryst_list", "delta_r_P", "q_phonon_cryst")};

// compute_delta_V_new
static auto const fun_thc_dV_new = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb &self,
       const std::string &Deltapsi_prefix,
       const std::string &Deltapsi_adj_prefix,
       nda::array<double, 1> const &q_pert_cryst) {
      return self.compute_delta_V_new(Deltapsi_prefix, Deltapsi_adj_prefix, q_pert_cryst);
    },
    "self", "Deltapsi_prefix", "Deltapsi_adj_prefix", "q_pert_cryst")};

static const auto doc_d_0 = fun_0.doc(R"DOC()DOC");
static const auto doc_d_1 = fun_1.doc(R"DOC()DOC");
static const auto doc_d_2 = fun_2.doc(R"DOC()DOC");
static const auto doc_d_3 = fun_3.doc(R"DOC()DOC");
static const auto doc_d_4 = fun_4.doc(R"DOC()DOC");
static const auto doc_d_5 = fun_5.doc(R"DOC()DOC");
static const auto doc_d_6 = fun_6.doc(R"DOC()DOC");
static const auto doc_d_7 = fun_7.doc(R"DOC()DOC");
static const auto doc_d_8 = fun_8.doc(R"DOC()DOC");
static const auto doc_d_9 = fun_9.doc(R"DOC()DOC");
static const auto doc_d_10 = fun_10.doc(R"DOC()DOC");
static const auto doc_d_11 = fun_11.doc(R"DOC()DOC");
static const auto doc_thc_DeltaXi = fun_thc_DeltaXi.doc(R"DOC(
Compute Ξ^q (self-consistent) and its linear response δΞ^q.

Parameters
----------
Deltapsi_prefix : str
    Per-k δψ files are {Deltapsi_prefix}_ik{ik+1}.hdf5 (QE format: MillerIndices + evc).
q_vec_cryst : np.ndarray of shape (3,)
    q-vector in crystal coordinates; must match an entry of Qpts_ibz.
delta_r_P : np.ndarray of shape (Np, 3), real
    Linear response of the interpolating points r_μ in Cartesian Bohr.
    Pass zeros for the fixed-r_μ case.

Returns
-------
tuple (Ξ^q, δΞ^q) of two np.ndarray, each shape (Np, nnr), complex
    Replicated on every rank. Unpack as ``Xi, dXi = thc.compute_delta_Xi(...)``.

Notes
-----
Current implementation restrictions: npol == 1, nkpts == nkpts_ibz
(no k-point symmetry). delta_r_P is restricted to real values;
complex Δr_μ (needed for q≠0 phonon perturbations) is not yet supported.
)DOC");

static const auto doc_thc_dV = fun_thc_dV.doc(R"DOC(
Compute δV^q_{μν}, the linear response of the THC Coulomb matrix.

    δV^q_{μν} = ∫∫ dr dr' [δξ^{-q}_μ(r) v(r,r') ξ^q_ν(r')
                          + ξ^{-q}_μ(r) v(r,r') δξ^q_ν(r')]

V^q itself is already available from the ordinary THC decomposition and is not
returned here. Arguments mirror compute_delta_Xi.

Parameters
----------
Deltapsi_prefix : str
    Per-k δψ files {Deltapsi_prefix}_ik{ik+1}.hdf5 (QE format).
q_vec_cryst : np.ndarray of shape (3,)
    q-vector in crystal coordinates.
delta_r_P : np.ndarray of shape (Np, 3), real
    Linear response of r_μ in Cartesian Bohr.

Returns
-------
np.ndarray of shape (Np, Np), complex
    δV^q. Replicated on every rank.
)DOC");

static const auto doc_thc_dV_batched = fun_thc_dV_batched.doc(R"DOC(
Batched variant of compute_delta_V: compute δV^q for all internal q in
q_vec_cryst_list, amortizing the expensive wfc/δψ I/O and IFFT across q.

Parameters
----------
Deltapsi_prefix : str
    Per-k δψ files {Deltapsi_prefix}_ik{ik+1}.hdf5 (QE format).
q_vec_cryst_list : np.ndarray of shape (nq, 3)
    List of internal q-vectors in crystal coordinates; each must lie on the
    BZ k-grid.
delta_r_P : np.ndarray of shape (Np, 3), real
    Linear response of r_μ in Cartesian Bohr.
q_phonon_cryst : np.ndarray of shape (3,), optional
    Phonon wavevector Q of δψ in crystal coordinates (defaults to Γ).

Returns
-------
np.ndarray of shape (nq, Np, Np), complex
    δV^{q_list[iq]}. Replicated on every rank.

Notes
-----
Stage A (wfc/δψ I/O + IFFT + T, δT build) runs once; stages B + C loop over q.
Memory cost: the internal per-k caches (T, δT on the full FFT mesh plus
δr·∇₂T on the pivot grid) are held in full on every rank. Dominant term is
T + δT: ≈ 2 · nspin · nkpts · Np · nnr · 16 bytes.
)DOC");

static const auto doc_thc_dV_new = fun_thc_dV_new.doc(R"DOC(
Linear response of V^q for a single phonon momentum Q (theory-note
convention, see docs/lr_thc_theory.txt). Returns δV^q for every internal q
on the BZ k-grid in a single call.

Unlike compute_delta_V[_batched], this entry point is correct for general
phonon Q ≠ 0: it carries the +Q and −Q wavefunction-response channels
separately, uses the shifted Ξ^{q+Q} on the right-hand side of the LR
normal equation, and applies the correct Coulomb factors v(|q+G|) and
v(|q+Q+G|) to the two pieces of δV. The pivot response (Δr_P) is ignored.

Parameters
----------
Deltapsi_prefix : str
    Per-k +Q δψ files. {Deltapsi_prefix}_ik{ik+1}.hdf5 stores
    δ^{+Q} ψ^{k_ik}.
Deltapsi_adj_prefix : str
    Per-k −Q δψ (adjoint) files. {Deltapsi_adj_prefix}_ik{ik+1}.hdf5 stores
    δ^{−Q} ψ^{k_ik + Q}.
q_pert_cryst : np.ndarray of shape (3,)
    Phonon wavevector Q in crystal coordinates.

Returns
-------
np.ndarray of shape (nq, Np, Np), complex
    δV^q at every internal q on the BZ k-grid (same order as
    mf.kpts_crystal()). Replicated on every rank.

Notes
-----
This is a serial reference implementation (no MPI distribution, no GPU,
no OpenMP). Restrictions: npol == 1, nkpts == nkpts_ibz.
)DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<coqui_py::ThcCoulomb>[] = {
    {"Np", (PyCFunction)c2py::pyfkw<fun_0>, METH_VARARGS | METH_KEYWORDS,
     doc_d_0.c_str()},
    {"init", (PyCFunction)c2py::pyfkw<fun_1>, METH_VARARGS | METH_KEYWORDS,
     doc_d_1.c_str()},
    {"initialized", (PyCFunction)c2py::pyfkw<fun_2>,
     METH_VARARGS | METH_KEYWORDS, doc_d_2.c_str()},
    {"mf", (PyCFunction)c2py::pyfkw<fun_3>, METH_VARARGS | METH_KEYWORDS,
     doc_d_3.c_str()},
    {"mpi", (PyCFunction)c2py::pyfkw<fun_4>, METH_VARARGS | METH_KEYWORDS,
     doc_d_4.c_str()},
    {"nbnd", (PyCFunction)c2py::pyfkw<fun_5>, METH_VARARGS | METH_KEYWORDS,
     doc_d_5.c_str()},
    {"nkpts", (PyCFunction)c2py::pyfkw<fun_6>, METH_VARARGS | METH_KEYWORDS,
     doc_d_6.c_str()},
    {"nkpts_ibz", (PyCFunction)c2py::pyfkw<fun_7>, METH_VARARGS | METH_KEYWORDS,
     doc_d_7.c_str()},
    {"nqpts", (PyCFunction)c2py::pyfkw<fun_8>, METH_VARARGS | METH_KEYWORDS,
     doc_d_8.c_str()},
    {"nqpts_ibz", (PyCFunction)c2py::pyfkw<fun_9>, METH_VARARGS | METH_KEYWORDS,
     doc_d_9.c_str()},
    {"nspin", (PyCFunction)c2py::pyfkw<fun_10>, METH_VARARGS | METH_KEYWORDS,
     doc_d_10.c_str()},
    {"nspin_in_basis", (PyCFunction)c2py::pyfkw<fun_11>,
     METH_VARARGS | METH_KEYWORDS, doc_d_11.c_str()},
    {"compute_delta_Xi", (PyCFunction)c2py::pyfkw<fun_thc_DeltaXi>,
     METH_VARARGS | METH_KEYWORDS, doc_thc_DeltaXi.c_str()},
    {"compute_delta_V", (PyCFunction)c2py::pyfkw<fun_thc_dV>,
     METH_VARARGS | METH_KEYWORDS, doc_thc_dV.c_str()},
    {"compute_delta_V_batched", (PyCFunction)c2py::pyfkw<fun_thc_dV_batched>,
     METH_VARARGS | METH_KEYWORDS, doc_thc_dV_batched.c_str()},
    {"compute_delta_V_new", (PyCFunction)c2py::pyfkw<fun_thc_dV_new>,
     METH_VARARGS | METH_KEYWORDS, doc_thc_dV_new.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

// ----- Method table ----

template <>
constinit PyGetSetDef c2py::tp_getset<coqui_py::ThcCoulomb>[] = {

    {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <>
const std::string c2py::tp_doc<coqui_py::ThcCoulomb> =
    R"DOC()DOC" + c2py::tp_ctor_doc<coqui_py::ThcCoulomb>;
template <>
inline constexpr auto c2py::tp_name<coqui_py::CholCoulomb> =
    "eri_module.CholCoulomb";
static auto init_1 = c2py::dispatcher_c_kw_t{
    c2py::c_constructor<coqui_py::CholCoulomb, const coqui_py::Mf &,
                        const std::string &>("mf", "chol_params")};
template <>
constexpr initproc c2py::tp_init<coqui_py::CholCoulomb> =
    c2py::pyfkw_constructor<init_1>;
template <>
const std::string c2py::tp_ctor_doc<coqui_py::CholCoulomb> =
    init_1.doc(R"DOC()DOC");
// mf
static auto const fun_12 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::CholCoulomb const &self) { return self.mf(); }, "self")};

// mpi
static auto const fun_13 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::CholCoulomb const &self) { return self.mpi(); }, "self")};

static const auto doc_d_12 = fun_12.doc(R"DOC()DOC");
static const auto doc_d_13 = fun_13.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<coqui_py::CholCoulomb>[] = {
    {"mf", (PyCFunction)c2py::pyfkw<fun_12>, METH_VARARGS | METH_KEYWORDS,
     doc_d_12.c_str()},
    {"mpi", (PyCFunction)c2py::pyfkw<fun_13>, METH_VARARGS | METH_KEYWORDS,
     doc_d_13.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

// ----- Method table ----

template <>
constinit PyGetSetDef c2py::tp_getset<coqui_py::CholCoulomb>[] = {

    {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <>
const std::string c2py::tp_doc<coqui_py::CholCoulomb> =
    R"DOC()DOC" + c2py::tp_ctor_doc<coqui_py::CholCoulomb>;

// ==================== module functions ====================

// run_isdf
static auto const fun_14 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &thc_params) {
      return coqui_py::run_isdf(mf, thc_params);
    },
    "mf", "thc_params")};

static const auto doc_d_14 = fun_14.doc(R"DOC()DOC");

// compute_delta_X
static auto const fun_15 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf,
       const std::string &Deltapsi_prefix,
       nda::array<long, 1> const& r_P,
       nda::array<double, 2> const& delta_r_P,
       nda::array<double, 1> const& q_vec_cryst,
       nda::array<long, 1> const& fft_grid) {
      return coqui_py::compute_delta_X(mf, Deltapsi_prefix,
                                        r_P, delta_r_P, q_vec_cryst, fft_grid);
    },
    "mf", "Deltapsi_prefix", "r_P", "delta_r_P", "q_vec_cryst", "fft_grid")};

static const auto doc_d_15 = fun_15.doc(R"DOC(
Compute DeltaX(k, P, m) = Deltapsi_mk(r_P) + grad psi_mk(r_P) . Delta_r_P

Deltapsi files are per-mode per-k: {Deltapsi_prefix}_ik{ik+1}.hdf5,
each containing MillerIndices (npw, 3) and evc (nbnd, 2*npw).

Parameters
----------
mf : Mf
    Mean-field object (for reading unperturbed wfc and system info)
Deltapsi_prefix : str
    File path prefix including mode. Files are {Deltapsi_prefix}_ik{ik+1}.hdf5.
    Example: "temp/Deltapsi_ibc_iq1_mode1" -> temp/Deltapsi_ibc_iq1_mode1_ik1.hdf5, ...
r_P : np.ndarray
    (nP,) FFT grid indices of interpolating points
delta_r_P : np.ndarray
    (nP, 3) grid point displacements in Cartesian Bohr
q_vec_cryst : np.ndarray
    (3,) q-vector in crystal coordinates
fft_grid : np.ndarray
    (3,) FFT grid dimensions matching the grid used for r_P
    (e.g. from the THC checkpoint fft_grid dataset)

Returns
-------
np.ndarray
    (nspin, nk_ibz, nP, nbnd) complex array
)DOC");

// compute_delta_X_adj
static auto const fun_16 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf,
       const std::string &Deltapsi_adj_prefix,
       nda::array<long, 1> const& r_P,
       nda::array<double, 2> const& delta_r_P,
       nda::array<double, 1> const& q_vec_cryst,
       nda::array<long, 1> const& fft_grid) {
      return coqui_py::compute_delta_X_adj(mf, Deltapsi_adj_prefix,
                                            r_P, delta_r_P, q_vec_cryst, fft_grid);
    },
    "mf", "Deltapsi_adj_prefix", "r_P", "delta_r_P", "q_vec_cryst", "fft_grid")};

static const auto doc_d_16 = fun_16.doc(R"DOC(
Compute the adjoint/minus-q DeltaX:
    DeltaX_adj(k+q, P, m) = [δ^q]† ψ_{m,k+q}(r_P)
                          + grad ψ_{m,k+q}(r_P) · delta_r_P

At index ik the returned array stores the quantity at k_{ik}+q, matching
the convention used by DeltaX_right in lr_thc_comm (accessed as
DeltaX_right[kpq_map(k)]).

Term 1 reads [δ^q]† ψ_{k+q} from adjoint-Sternheimer HDF5 files:
    {Deltapsi_adj_prefix}_ik{ik+1}.hdf5
Term 2 uses unperturbed ψ at k+q (looked up via a local k→k+q map) and the
standard i(k+q+G) gradient. Requires the mean field to be on a full-BZ grid
(nkpts_ibz == nkpts), matching the IBC consumer.

Parameters
----------
mf : Mf
    Mean-field object (for reading unperturbed wfc and system info)
Deltapsi_adj_prefix : str
    File path prefix for adjoint-Sternheimer files.
    Example: "temp/dpsi_adj_iq1_mode1" -> temp/dpsi_adj_iq1_mode1_ik1.hdf5, ...
r_P : np.ndarray
    (nP,) FFT grid indices of interpolating points
delta_r_P : np.ndarray
    (nP, 3) grid point displacements in Cartesian Bohr. Interpreted as the
    +q displacement; under the assumption δ^{-q} = conj(δ^q) and real
    displacements, no conjugation is needed on the caller side.
q_vec_cryst : np.ndarray
    (3,) +q vector in crystal coordinates
fft_grid : np.ndarray
    (3,) FFT grid dimensions matching the grid used for r_P

Returns
-------
np.ndarray
    (nspin, nk_ibz, nP, nbnd) complex array
)DOC");

//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
    {"run_isdf", (PyCFunction)c2py::pyfkw<fun_14>, METH_VARARGS | METH_KEYWORDS,
     doc_d_14.c_str()},
    {"compute_delta_X", (PyCFunction)c2py::pyfkw<fun_15>, METH_VARARGS | METH_KEYWORDS,
     doc_d_15.c_str()},
    {"compute_delta_X_adj", (PyCFunction)c2py::pyfkw<fun_16>, METH_VARARGS | METH_KEYWORDS,
     doc_d_16.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "eri_module",                                  /* name of module */
    R"RAWDOC(Interaction module for CoQui)RAWDOC", /* module documentation, may
                                                      be NULL */
    -1, /* size of per-interpreter state of the module, or -1 if the module
           keeps state in global variables. */
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *
PyInit_eri_module() {

  if (not c2py::check_python_version("eri_module"))
    return NULL;

  // import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  PyObject *m;

  if (PyType_Ready(&c2py::wrap_pytype<c2py::py_range>) < 0)
    return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<coqui_py::ThcCoulomb>) < 0)
    return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<coqui_py::CholCoulomb>) < 0)
    return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL)
    return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] =
      &c2py::wrap_pytype<c2py::py_range>;
  c2py::add_type_object_to_main<coqui_py::ThcCoulomb>("ThcCoulomb", m,
                                                      conv_table);
  c2py::add_type_object_to_main<coqui_py::CholCoulomb>("CholCoulomb", m,
                                                       conv_table);

  return m;
}
#endif
// CLAIR_WRAP_GEN
