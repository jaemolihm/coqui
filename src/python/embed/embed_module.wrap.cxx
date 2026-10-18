
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

// ==================== enums =====================

// ==================== module classes =====================

// ==================== module functions ====================

// dmft_embed
static auto const _c2py_fun_0 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &embed_params,
       const nda::basic_array<
           std::complex<double>, 5, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &projector_ksIai,
       const nda::basic_array<
           long, 3, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &band_window,
       const nda::basic_array<
           double, 2, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &kpts_crys,
       std::optional<std::map<
           std::string,
           nda::basic_array<std::complex<double>, 4, nda::C_layout, 'A',
                            nda::heap_basic<nda::mem::mallocator<
                                nda::mem::AddressSpace::Host>>>>>
           local_hf_potentials,
       std::optional<std::map<
           std::string,
           nda::basic_array<std::complex<double>, 5, nda::C_layout, 'A',
                            nda::heap_basic<nda::mem::mallocator<
                                nda::mem::AddressSpace::Host>>>>>
           local_selfenergies) {
      return coqui_py::dmft_embed(mf, embed_params, projector_ksIai,
                                  band_window, kpts_crys, local_hf_potentials,
                                  local_selfenergies);
    },
    "mf", "embed_params", "projector_ksIai", "band_window", "kpts_crys",
    "local_hf_potentials", "local_selfenergies")};

// dmft_embed_with_projector_from_h5
static auto const _c2py_fun_1 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &embed_params,
       std::optional<std::map<
           std::string,
           nda::basic_array<std::complex<double>, 4, nda::C_layout, 'A',
                            nda::heap_basic<nda::mem::mallocator<
                                nda::mem::AddressSpace::Host>>>>>
           local_hf_potentials,
       std::optional<std::map<
           std::string,
           nda::basic_array<std::complex<double>, 5, nda::C_layout, 'A',
                            nda::heap_basic<nda::mem::mallocator<
                                nda::mem::AddressSpace::Host>>>>>
           local_selfenergies) {
      return coqui_py::dmft_embed_with_projector_from_h5(
          mf, embed_params, local_hf_potentials, local_selfenergies);
    },
    "mf", "embed_params", "local_hf_potentials", "local_selfenergies")};

// downfold_1e
static auto const _c2py_fun_2 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &df_params) {
      return coqui_py::downfold_1e(mf, df_params);
    },
    "mf", "df_params")};

// downfold_2e
static auto const _c2py_fun_3 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](coqui_py::ThcCoulomb &eri, const std::string &df_params,
       std::optional<std::map<
           std::string,
           nda::basic_array<std::complex<double>, 5, nda::C_layout, 'A',
                            nda::heap_basic<nda::mem::mallocator<
                                nda::mem::AddressSpace::Host>>>>>
           local_polarizabilities) {
      return coqui_py::downfold_2e<coqui_py::ThcCoulomb>(
          eri, df_params, local_polarizabilities);
    },
    "eri", "df_params", "local_polarizabilities")};

// downfold_coulomb
static auto const _c2py_fun_4 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](coqui_py::ThcCoulomb &eri, const std::string &df_params,
       const nda::basic_array<
           std::complex<double>, 5, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &projector_ksIai,
       const nda::basic_array<
           long, 3, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &band_window,
       const nda::basic_array<
           double, 2, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &kpts_crys,
       std::optional<std::map<
           std::string,
           nda::basic_array<std::complex<double>, 5, nda::C_layout, 'A',
                            nda::heap_basic<nda::mem::mallocator<
                                nda::mem::AddressSpace::Host>>>>>
           local_polarizabilities) {
      return coqui_py::downfold_coulomb<coqui_py::ThcCoulomb>(
          eri, df_params, projector_ksIai, band_window, kpts_crys,
          local_polarizabilities);
    },
    "eri", "df_params", "projector_ksIai", "band_window", "kpts_crys",
    "local_polarizabilities")};

// downfold_coulomb_with_projector_from_h5
static auto const _c2py_fun_5 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](coqui_py::ThcCoulomb &eri, const std::string &df_params,
       std::optional<std::map<
           std::string,
           nda::basic_array<std::complex<double>, 5, nda::C_layout, 'A',
                            nda::heap_basic<nda::mem::mallocator<
                                nda::mem::AddressSpace::Host>>>>>
           local_polarizabilities) {
      return coqui_py::downfold_coulomb_with_projector_from_h5<
          coqui_py::ThcCoulomb>(eri, df_params, local_polarizabilities);
    },
    "eri", "df_params", "local_polarizabilities")};

// downfold_gloc
static auto const _c2py_fun_6 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &df_params,
       const nda::basic_array<
           std::complex<double>, 5, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &projector_ksIai,
       const nda::basic_array<
           long, 3, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &band_window,
       const nda::basic_array<
           double, 2, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &kpts_crys) {
      return coqui_py::downfold_gloc(mf, df_params, projector_ksIai,
                                     band_window, kpts_crys);
    },
    "mf", "df_params", "projector_ksIai", "band_window", "kpts_crys")};

// downfold_gloc_with_projector_from_h5
static auto const _c2py_fun_7 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &df_params) {
      return coqui_py::downfold_gloc_with_projector_from_h5(mf, df_params);
    },
    "mf", "df_params")};

// lr_downfold_coulomb
static auto const _c2py_fun_8 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](coqui_py::ThcCoulomb &eri, const std::string &df_params,
       const nda::basic_array<
           std::complex<double>, 5, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &projector_ksIai,
       const nda::basic_array<
           long, 3, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &band_window,
       const nda::basic_array<
           double, 2, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &kpts_crys,
       std::optional<nda::basic_array<
           std::complex<double>, 4, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>> DeltaW_xqPQ,
       const nda::basic_array<
           double, 1, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &q_vec) {
      return coqui_py::lr_downfold_coulomb<coqui_py::ThcCoulomb>(
          eri, df_params, projector_ksIai, band_window, kpts_crys, DeltaW_xqPQ, q_vec);
    },
    "eri", "df_params", "projector_ksIai", "band_window", "kpts_crys",
    "DeltaW_xqPQ", "q_vec")};

// lr_downfold_gloc
static auto const _c2py_fun_9 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &df_params,
       const nda::basic_array<
           std::complex<double>, 5, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &projector_ksIai,
       const nda::basic_array<
           long, 3, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &band_window,
       const nda::basic_array<
           double, 2, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &kpts_crys,
       std::optional<nda::basic_array<
           std::complex<double>, 5, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>> DeltaG_tskij,
       const nda::basic_array<
           double, 1, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &q_vec) {
      return coqui_py::lr_downfold_gloc(mf, df_params, projector_ksIai,
                                        band_window, kpts_crys, DeltaG_tskij, q_vec);
    },
    "mf", "df_params", "projector_ksIai", "band_window", "kpts_crys",
    "DeltaG_tskij", "q_vec")};

// eval_gw_dc_t
static auto const _c2py_fun_10 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const nda::array<std::complex<double>, 4> &G_tsab,
       const nda::array<std::complex<double>, 5> &W_tabcd) {
      return coqui_py::eval_gw_dc_t(G_tsab, W_tabcd);
    },
    "G_tsab", "W_tabcd")};

// lr_eval_gw_dc_t
static auto const _c2py_fun_11 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const nda::array<std::complex<double>, 4> &G_tsab,
       const nda::array<std::complex<double>, 4> &dG_tsab,
       const nda::array<std::complex<double>, 5> &W_tabcd,
       const nda::array<std::complex<double>, 5> &dW_tabcd) {
      return coqui_py::lr_eval_gw_dc_t(G_tsab, dG_tsab, W_tabcd, dW_tabcd);
    },
    "G_tsab", "dG_tsab", "W_tabcd", "dW_tabcd")};

// eval_pi_rpa_pair
static auto const _c2py_fun_12 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const nda::array<std::complex<double>, 4> &G1_tsab,
       const nda::array<std::complex<double>, 4> &G2_tsab) {
      return coqui_py::eval_pi_rpa_pair(G1_tsab, G2_tsab);
    },
    "G1_tsab", "G2_tsab")};

// eval_pi_rpa_density_pair
static auto const _c2py_fun_13 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const nda::array<std::complex<double>, 4> &G1_tsab,
       const nda::array<std::complex<double>, 4> &G2_tsab) {
      return coqui_py::eval_pi_rpa_density_pair(G1_tsab, G2_tsab);
    },
    "G1_tsab", "G2_tsab")};

// lr_eval_pi_rpa
static auto const _c2py_fun_14 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const nda::array<std::complex<double>, 4> &G_tsab,
       const nda::array<std::complex<double>, 4> &dG_tsab) {
      return coqui_py::lr_eval_pi_rpa(G_tsab, dG_tsab);
    },
    "G_tsab", "dG_tsab")};

// lr_eval_pi_rpa_density
static auto const _c2py_fun_15 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const nda::array<std::complex<double>, 4> &G_tsab,
       const nda::array<std::complex<double>, 4> &dG_tsab) {
      return coqui_py::lr_eval_pi_rpa_density(G_tsab, dG_tsab);
    },
    "G_tsab", "dG_tsab")};

// chemistry_to_product_basis
static auto const _c2py_fun_16 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const nda::array<std::complex<double>, 4> &A) {
      return coqui_py::chemistry_to_product_basis<4>(A);
    },
    "A"), c2py::cfun(
    [](const nda::array<std::complex<double>, 5> &A) {
      return coqui_py::chemistry_to_product_basis<5>(A);
    },
    "A")};

// density_density_to_product_basis
static auto const _c2py_fun_17 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const nda::array<std::complex<double>, 2> &A) {
      return coqui_py::density_density_to_product_basis<2>(A);
    },
    "A"), c2py::cfun(
    [](const nda::array<std::complex<double>, 3> &A) {
      return coqui_py::density_density_to_product_basis<3>(A);
    },
    "A")};

// duplicate_spin
static auto const _c2py_fun_18 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const nda::array<std::complex<double>, 4> &A,
       int axis) {
      return coqui_py::duplicate_spin<4>(A, axis);
    },
    "A", "axis"), c2py::cfun(
    [](const nda::array<std::complex<double>, 5> &A,
       int axis) {
      return coqui_py::duplicate_spin<5>(A, axis);
    },
    "A", "axis")};

// spin_average
static auto const _c2py_fun_19 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const nda::array<std::complex<double>, 4> &A,
       int axis) {
      return coqui_py::spin_average<4>(A, axis);
    },
    "A", "axis"), c2py::cfun(
    [](const nda::array<std::complex<double>, 5> &A,
       int axis) {
      return coqui_py::spin_average<5>(A, axis);
    },
    "A", "axis")};

// extract_impurity_blocks
static auto const _c2py_fun_20 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const std::vector<nda::array<std::complex<double>, 2>> &X_sigma,
       const nda::array<long, 1> &alpha_offset,
       const nda::array<long, 1> &alpha_size,
       const nda::array<long, 3> &psi,
       const nda::array<long, 1> &n_gamma,
       const std::vector<std::vector<long>> &imp_block_dim) {
      return coqui_py::extract_impurity_blocks<2>(X_sigma, alpha_offset, alpha_size, psi, n_gamma, imp_block_dim);
    },
    "X_sigma", "alpha_offset", "alpha_size", "psi", "n_gamma", "imp_block_dim"), c2py::cfun(
    [](const std::vector<nda::array<std::complex<double>, 3>> &X_sigma,
       const nda::array<long, 1> &alpha_offset,
       const nda::array<long, 1> &alpha_size,
       const nda::array<long, 3> &psi,
       const nda::array<long, 1> &n_gamma,
       const std::vector<std::vector<long>> &imp_block_dim) {
      return coqui_py::extract_impurity_blocks<3>(X_sigma, alpha_offset, alpha_size, psi, n_gamma, imp_block_dim);
    },
    "X_sigma", "alpha_offset", "alpha_size", "psi", "n_gamma", "imp_block_dim"), c2py::cfun(
    [](const std::vector<nda::array<std::complex<double>, 4>> &X_sigma,
       const nda::array<long, 1> &alpha_offset,
       const nda::array<long, 1> &alpha_size,
       const nda::array<long, 3> &psi,
       const nda::array<long, 1> &n_gamma,
       const std::vector<std::vector<long>> &imp_block_dim) {
      return coqui_py::extract_impurity_blocks<4>(X_sigma, alpha_offset, alpha_size, psi, n_gamma, imp_block_dim);
    },
    "X_sigma", "alpha_offset", "alpha_size", "psi", "n_gamma", "imp_block_dim"), c2py::cfun(
    [](const std::vector<nda::array<std::complex<double>, 5>> &X_sigma,
       const nda::array<long, 1> &alpha_offset,
       const nda::array<long, 1> &alpha_size,
       const nda::array<long, 3> &psi,
       const nda::array<long, 1> &n_gamma,
       const std::vector<std::vector<long>> &imp_block_dim) {
      return coqui_py::extract_impurity_blocks<5>(X_sigma, alpha_offset, alpha_size, psi, n_gamma, imp_block_dim);
    },
    "X_sigma", "alpha_offset", "alpha_size", "psi", "n_gamma", "imp_block_dim")};

// embed_impurity_blocks
static auto const _c2py_fun_21 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const std::vector<std::vector<nda::array<std::complex<double>, 2>>> &blocks,
       const nda::array<long, 1> &alpha_offset,
       const nda::array<long, 1> &alpha_size,
       const nda::array<long, 3> &psi,
       const nda::array<long, 1> &n_gamma,
       const std::vector<std::vector<long>> &imp_block_dim) {
      return coqui_py::embed_impurity_blocks<2>(blocks, alpha_offset, alpha_size, psi, n_gamma, imp_block_dim);
    },
    "blocks", "alpha_offset", "alpha_size", "psi", "n_gamma", "imp_block_dim"), c2py::cfun(
    [](const std::vector<std::vector<nda::array<std::complex<double>, 3>>> &blocks,
       const nda::array<long, 1> &alpha_offset,
       const nda::array<long, 1> &alpha_size,
       const nda::array<long, 3> &psi,
       const nda::array<long, 1> &n_gamma,
       const std::vector<std::vector<long>> &imp_block_dim) {
      return coqui_py::embed_impurity_blocks<3>(blocks, alpha_offset, alpha_size, psi, n_gamma, imp_block_dim);
    },
    "blocks", "alpha_offset", "alpha_size", "psi", "n_gamma", "imp_block_dim"), c2py::cfun(
    [](const std::vector<std::vector<nda::array<std::complex<double>, 4>>> &blocks,
       const nda::array<long, 1> &alpha_offset,
       const nda::array<long, 1> &alpha_size,
       const nda::array<long, 3> &psi,
       const nda::array<long, 1> &n_gamma,
       const std::vector<std::vector<long>> &imp_block_dim) {
      return coqui_py::embed_impurity_blocks<4>(blocks, alpha_offset, alpha_size, psi, n_gamma, imp_block_dim);
    },
    "blocks", "alpha_offset", "alpha_size", "psi", "n_gamma", "imp_block_dim"), c2py::cfun(
    [](const std::vector<std::vector<nda::array<std::complex<double>, 5>>> &blocks,
       const nda::array<long, 1> &alpha_offset,
       const nda::array<long, 1> &alpha_size,
       const nda::array<long, 3> &psi,
       const nda::array<long, 1> &n_gamma,
       const std::vector<std::vector<long>> &imp_block_dim) {
      return coqui_py::embed_impurity_blocks<5>(blocks, alpha_offset, alpha_size, psi, n_gamma, imp_block_dim);
    },
    "blocks", "alpha_offset", "alpha_size", "psi", "n_gamma", "imp_block_dim")};

// eval_gw_dc_cells
static auto const _c2py_fun_22 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const nda::array<std::complex<double>, 4> &G_tsab,
       const nda::array<std::complex<double>, 5> &W_tabcd,
       const nda::array<long, 1> &cell_offset,
       const nda::array<long, 1> &cell_size,
       const nda::array<long, 1> &orb_block) {
      return coqui_py::eval_gw_dc_cells(G_tsab, W_tabcd, cell_offset, cell_size, orb_block);
    },
    "G_tsab", "W_tabcd", "cell_offset", "cell_size", "orb_block")};

// lr_eval_gw_dc_cells
static auto const _c2py_fun_23 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const nda::array<std::complex<double>, 4> &G_tsab,
       const nda::array<std::complex<double>, 4> &dG_tsab,
       const nda::array<std::complex<double>, 5> &W_tabcd,
       const nda::array<std::complex<double>, 5> &dW_tabcd,
       const nda::array<long, 1> &cell_offset,
       const nda::array<long, 1> &cell_size,
       const nda::array<long, 1> &orb_block) {
      return coqui_py::lr_eval_gw_dc_cells(G_tsab, dG_tsab, W_tabcd, dW_tabcd, cell_offset, cell_size, orb_block);
    },
    "G_tsab", "dG_tsab", "W_tabcd", "dW_tabcd", "cell_offset", "cell_size", "orb_block")};

// eval_pi_rpa_cells
static auto const _c2py_fun_24 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const nda::array<std::complex<double>, 4> &G_tsab,
       const nda::array<long, 1> &cell_offset,
       const nda::array<long, 1> &cell_size,
       const nda::array<long, 1> &orb_block) {
      return coqui_py::eval_pi_rpa_cells(G_tsab, cell_offset, cell_size, orb_block);
    },
    "G_tsab", "cell_offset", "cell_size", "orb_block")};

// lr_eval_pi_rpa_cells
static auto const _c2py_fun_25 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const nda::array<std::complex<double>, 4> &G_tsab,
       const nda::array<std::complex<double>, 4> &dG_tsab,
       const nda::array<long, 1> &cell_offset,
       const nda::array<long, 1> &cell_size,
       const nda::array<long, 1> &orb_block) {
      return coqui_py::lr_eval_pi_rpa_cells(G_tsab, dG_tsab, cell_offset, cell_size, orb_block);
    },
    "G_tsab", "dG_tsab", "cell_offset", "cell_size", "orb_block")};

static const auto _c2py_doc_0 = _c2py_fun_0.doc(R"DOC()DOC");
static const auto _c2py_doc_1 = _c2py_fun_1.doc(R"DOC()DOC");
static const auto _c2py_doc_2 = _c2py_fun_2.doc(R"DOC()DOC");
static const auto _c2py_doc_3 = _c2py_fun_3.doc(R"DOC()DOC");
static const auto _c2py_doc_4 = _c2py_fun_4.doc(R"DOC()DOC");
static const auto _c2py_doc_5 = _c2py_fun_5.doc(R"DOC()DOC");
static const auto _c2py_doc_6 = _c2py_fun_6.doc(R"DOC()DOC");
static const auto _c2py_doc_7 = _c2py_fun_7.doc(R"DOC()DOC");
static const auto _c2py_doc_8 = _c2py_fun_8.doc(R"DOC()DOC");
static const auto _c2py_doc_9 = _c2py_fun_9.doc(R"DOC()DOC");
static const auto _c2py_doc_10 = _c2py_fun_10.doc(R"DOC()DOC");
static const auto _c2py_doc_11 = _c2py_fun_11.doc(R"DOC()DOC");
static const auto _c2py_doc_12 = _c2py_fun_12.doc(R"DOC()DOC");
static const auto _c2py_doc_13 = _c2py_fun_13.doc(R"DOC()DOC");
static const auto _c2py_doc_14 = _c2py_fun_14.doc(R"DOC()DOC");
static const auto _c2py_doc_15 = _c2py_fun_15.doc(R"DOC()DOC");
static const auto _c2py_doc_16 = _c2py_fun_16.doc(R"DOC()DOC");
static const auto _c2py_doc_17 = _c2py_fun_17.doc(R"DOC()DOC");
static const auto _c2py_doc_18 = _c2py_fun_18.doc(R"DOC()DOC");
static const auto _c2py_doc_19 = _c2py_fun_19.doc(R"DOC()DOC");
static const auto _c2py_doc_20 = _c2py_fun_20.doc(R"DOC()DOC");
static const auto _c2py_doc_21 = _c2py_fun_21.doc(R"DOC()DOC");
static const auto _c2py_doc_22 = _c2py_fun_22.doc(R"DOC()DOC");
static const auto _c2py_doc_23 = _c2py_fun_23.doc(R"DOC()DOC");
static const auto _c2py_doc_24 = _c2py_fun_24.doc(R"DOC()DOC");
static const auto _c2py_doc_25 = _c2py_fun_25.doc(R"DOC()DOC");
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
    {"dmft_embed", (PyCFunction)c2py::pyfkw<_c2py_fun_0>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_0.c_str()},
    {"dmft_embed_with_projector_from_h5", (PyCFunction)c2py::pyfkw<_c2py_fun_1>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_1.c_str()},
    {"downfold_1e", (PyCFunction)c2py::pyfkw<_c2py_fun_2>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_2.c_str()},
    {"downfold_2e", (PyCFunction)c2py::pyfkw<_c2py_fun_3>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_3.c_str()},
    {"downfold_coulomb", (PyCFunction)c2py::pyfkw<_c2py_fun_4>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_4.c_str()},
    {"downfold_coulomb_with_projector_from_h5",
     (PyCFunction)c2py::pyfkw<_c2py_fun_5>, METH_VARARGS | METH_KEYWORDS,
     _c2py_doc_5.c_str()},
    {"downfold_gloc", (PyCFunction)c2py::pyfkw<_c2py_fun_6>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_6.c_str()},
    {"downfold_gloc_with_projector_from_h5",
     (PyCFunction)c2py::pyfkw<_c2py_fun_7>, METH_VARARGS | METH_KEYWORDS,
     _c2py_doc_7.c_str()},
    {"lr_downfold_coulomb", (PyCFunction)c2py::pyfkw<_c2py_fun_8>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_8.c_str()},
    {"lr_downfold_gloc", (PyCFunction)c2py::pyfkw<_c2py_fun_9>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_9.c_str()},
    {"eval_gw_dc_t", (PyCFunction)c2py::pyfkw<_c2py_fun_10>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_10.c_str()},
    {"lr_eval_gw_dc_t", (PyCFunction)c2py::pyfkw<_c2py_fun_11>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_11.c_str()},
    {"eval_pi_rpa_pair", (PyCFunction)c2py::pyfkw<_c2py_fun_12>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_12.c_str()},
    {"eval_pi_rpa_density_pair", (PyCFunction)c2py::pyfkw<_c2py_fun_13>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_13.c_str()},
    {"lr_eval_pi_rpa", (PyCFunction)c2py::pyfkw<_c2py_fun_14>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_14.c_str()},
    {"lr_eval_pi_rpa_density", (PyCFunction)c2py::pyfkw<_c2py_fun_15>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_15.c_str()},
    {"chemistry_to_product_basis", (PyCFunction)c2py::pyfkw<_c2py_fun_16>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_16.c_str()},
    {"density_density_to_product_basis", (PyCFunction)c2py::pyfkw<_c2py_fun_17>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_17.c_str()},
    {"duplicate_spin", (PyCFunction)c2py::pyfkw<_c2py_fun_18>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_18.c_str()},
    {"spin_average", (PyCFunction)c2py::pyfkw<_c2py_fun_19>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_19.c_str()},
    {"extract_impurity_blocks", (PyCFunction)c2py::pyfkw<_c2py_fun_20>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_20.c_str()},
    {"embed_impurity_blocks", (PyCFunction)c2py::pyfkw<_c2py_fun_21>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_21.c_str()},
    {"eval_gw_dc_cells", (PyCFunction)c2py::pyfkw<_c2py_fun_22>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_22.c_str()},
    {"lr_eval_gw_dc_cells", (PyCFunction)c2py::pyfkw<_c2py_fun_23>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_23.c_str()},
    {"eval_pi_rpa_cells", (PyCFunction)c2py::pyfkw<_c2py_fun_24>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_24.c_str()},
    {"lr_eval_pi_rpa_cells", (PyCFunction)c2py::pyfkw<_c2py_fun_25>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_25.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "embed_module",                              /* name of module */
    R"RAWDOC(Embedding module for CoQui)RAWDOC", /* module documentation, may be
                                                    NULL */
    -1, /* size of per-interpreter state of the module, or -1 if the module
           keeps state in global variables. */
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *
PyInit_embed_module() {

  if (not c2py::check_python_version("embed_module"))
    return NULL;

  // import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  PyObject *m;

  if (PyType_Ready(&c2py::wrap_pytype<c2py::py_range>) < 0)
    return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL)
    return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] =
      &c2py::wrap_pytype<c2py::py_range>;
#define _add_type(T, N) c2py::add_type_object_to_main<T>(N, m, conv_table)

#undef _add_type

  return m;
}
#endif
// CLAIR_WRAP_GEN
