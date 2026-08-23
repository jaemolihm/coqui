/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */


#include "chkpt_utils.h"
#include "utilities/check.hpp"

#include <chrono>
#include <filesystem>

namespace methods {
  namespace chkpt {

template<typename communicator_t>
void write_metadata(communicator_t &comm, const mf::MF &mf, const imag_axes_ft::IAFT &ft,
                     const sArray_t<Array_view_4D_t> &sH0_skij,
                     const sArray_t<Array_view_4D_t> &sS_skij,
                     std::string output) {
  if (comm.root()) {
    std::string filename = output + ".mbpt.h5";
    h5::file file(filename, 'w');
    h5::group grp(file);

    auto sys_grp = grp.create_group("system");
    h5::h5_write(sys_grp, "number_of_spins", mf.nspin());
    h5::h5_write(sys_grp, "number_of_kpoints", mf.nkpts());
    h5::h5_write(sys_grp, "number_of_kpoints_ibz", mf.nkpts_ibz());
    h5::h5_write(sys_grp, "number_of_orbitals", mf.nbnd());
    h5::h5_write(sys_grp, "number_of_polarizations", mf.npol());
    h5::h5_write(sys_grp, "volume", mf.volume());
    h5::h5_write(sys_grp, "madelung", mf.madelung());
    nda::h5_write(sys_grp, "kp_grid", mf.kp_grid(), false);
    nda::h5_write(sys_grp, "kpoints", mf.kpts(), false);
    nda::h5_write(sys_grp, "kpoints_crys", mf.kpts_crystal(), false);
    nda::h5_write(sys_grp, "k_weight", mf.k_weight(), false);
    nda::h5_write(sys_grp, "k_trev", mf.kp_trev(), false);
    nda::h5_write(sys_grp, "kp_to_ibz", mf.kp_to_ibz(), false);
    nda::h5_write(sys_grp, "qpoints", mf.Qpts(), false);
    nda::h5_write(sys_grp, "qk_to_k2", mf.qk_to_k2(), false);
    nda::h5_write(sys_grp, "qminus", mf.qminus(), false);
    auto sH0_loc = sH0_skij.local();
    nda::h5_write(sys_grp, "H0_skij", sH0_loc, false);
    auto sSloc = sS_skij.local();
    nda::h5_write(sys_grp, "S_skij", sSloc, false);

    auto mf_grp = grp.create_group("mean_field");
    nda::h5_write(mf_grp, "eigvals", mf.eigval(), false);

    auto iaft_grp = grp.create_group("imaginary_fourier_transform");
    std::string iaft_basis = imag_axes_ft::basis_enum_to_string(ft.basis());
    h5::h5_write(iaft_grp, "basis", iaft_basis);
    h5::h5_write(iaft_grp, "prec", ft.prec());
    h5::h5_write(iaft_grp, "eps", ft.eps());
    h5::h5_write(iaft_grp, "beta", ft.beta());
    h5::h5_write(iaft_grp, "wmax", ft.wmax());
    h5::h5_write(iaft_grp, "lambda", ft.lambda());
    auto tau_grp = iaft_grp.create_group("tau_mesh");
    nda::h5_write(tau_grp, "fermion", ft.tau_mesh(), false);
    nda::h5_write(tau_grp, "boson", ft.tau_mesh_b(), false);
    auto iwn_grp = iaft_grp.create_group("iwn_mesh");
    nda::h5_write(iwn_grp, "fermion", ft.wn_mesh(), false);
    nda::h5_write(iwn_grp, "boson", ft.wn_mesh_b(), false);
  }
  comm.barrier();
}

template<typename communicator_t, typename X_t, typename Xt_t>
void dump_scf(communicator_t &comm, long iter,
              const X_t &Dm, const Xt_t &G,
              const X_t &F, const Xt_t &Sigma,
              double mu, std::string output,
              std::string input_grp, long input_iter,
              bool slim, const X_t *K) {
  if (comm.root()) {
    using clock_t = std::chrono::steady_clock;
    auto elapsed = [](clock_t::time_point a, clock_t::time_point b) {
      return std::chrono::duration<double>(b - a).count();
    };
    double t_open = 0, t_G = 0, t_Sigma = 0, t_small = 0, t_close = 0;
    // In a slim iteration, omit the frequency-dependent G and Sigma datasets.
    // The explicit metadata flag lets readers distinguish this from old files
    // where a missing Sigma dataset meant an exactly-zero Sigma (e.g. HF).
    auto Sigma_loc = Sigma.local();
    const char *sigma_note = slim ? " (skipped, slim iter)" : "";

    std::string filename = output + ".mbpt.h5";
    std::string iter_grp_name = "iter" + std::to_string(iter);
    auto t0 = clock_t::now();
    {
      h5::file file(filename, 'a');
      h5::group grp(file);
      auto scf_grp = (grp.has_subgroup("scf"))? grp.open_group("scf") : grp.create_group("scf");
      auto iter_grp = (scf_grp.has_subgroup(iter_grp_name) )?
          scf_grp.open_group(iter_grp_name) : scf_grp.create_group(iter_grp_name);

      if (input_iter==-1) input_iter = iter-1;

      h5::h5_write(scf_grp, "final_iter", iter);
      h5::h5_write(iter_grp, "greens_func_source", input_grp);
      h5::h5_write(iter_grp, "greens_func_iteration", input_iter);
      auto t1 = clock_t::now(); t_open = elapsed(t0, t1);

      if (!slim)
        nda::h5_write(iter_grp, "G_tskij", G.local(), false);
      auto t2 = clock_t::now(); t_G = elapsed(t1, t2);

      if (!slim)
        nda::h5_write(iter_grp, "Sigma_tskij", Sigma_loc, false);
      auto t3 = clock_t::now(); t_Sigma = elapsed(t2, t3);

      nda::h5_write(iter_grp, "F_skij", F.local(), false);
      // Opt-in exchange-only Fock K (F = J + K). Written only when requested so
      // the default checkpoint layout is unchanged.
      if (K) nda::h5_write(iter_grp, "K_skij", K->local(), false);
      nda::h5_write(iter_grp, "Dm_skij", Dm.local(), false);
      h5::h5_write(iter_grp, "mu", mu);
      h5::h5_write(iter_grp, "Sigma_not_stored", slim);
      auto t4 = clock_t::now(); t_small = elapsed(t3, t4);
    } // file flush + close (the ceph-bound cost happens here)
    t_close = elapsed(t0, clock_t::now()) - t_open - t_G - t_Sigma - t_small;

    app_log(2, "  dump_scf write breakdown (s): open/meta {:.3f}, G {:.3f}{}, Sigma {:.3f}{}, "
               "F+Dm {:.3f}, flush/close {:.3f}, total {:.3f}",
            t_open, t_G, slim ? " (skipped, slim iter)" : "",
            t_Sigma, sigma_note,
            t_small, t_close, t_open + t_G + t_Sigma + t_small + t_close);
  }
  comm.barrier();
}

template<typename communicator_t, typename X_4D_t, typename X_3D_t>
void dump_scf(communicator_t &comm, long iter,
              const X_4D_t &Dm_skij, const X_4D_t &Heff_skij,
              const X_4D_t &MO_skia, const X_3D_t &E_ska,
              double mu, std::string output) {
  if (comm.root()) {
    std::string filename = output + ".mbpt.h5";
    std::string iter_grp_name = "iter" + std::to_string(iter);
    h5::file file(filename, 'a');
    h5::group grp(file);
    auto scf_grp = (grp.has_subgroup("scf"))? grp.open_group("scf") : grp.create_group("scf");
    auto iter_grp = (scf_grp.has_subgroup(iter_grp_name) )?
                    scf_grp.open_group(iter_grp_name) : scf_grp.create_group(iter_grp_name);

    h5::h5_write(scf_grp, "final_iter", iter);
    nda::h5_write(iter_grp, "Dm_skij", Dm_skij.local(), false);
    nda::h5_write(iter_grp, "Heff_skij", Heff_skij.local(), false);
    nda::h5_write(iter_grp, "MO_skia", MO_skia.local(), false);
    nda::h5_write(iter_grp, "E_ska", E_ska.local(), false);
    h5::h5_write(iter_grp, "mu", mu);
  }
  comm.barrier();
}

template<typename X_t, typename Xt_t>
long read_scf(mpi3::shared_communicator node_comm,
              X_t &F, Xt_t &Sigma, double &mu,
              std::string output, std::string h5_grp, long iter) {
  if (node_comm.root()) {
    std::string filename = output + ".mbpt.h5";
    h5::file file(filename, 'r');

    auto scf_grp = h5::group(file).open_group(h5_grp);
    if (iter == -1) h5::h5_read(scf_grp, "final_iter", iter);

    auto iter_grp = scf_grp.open_group("iter"+std::to_string(iter));
    auto Sloc = Sigma.local();
    auto Floc = F.local();
    if (iter_grp.has_dataset("F_skij")) {
      // checkpoint from a dyson scf
      nda::h5_read(iter_grp, "F_skij", Floc);
      bool sigma_not_stored = false;
      if (iter_grp.has_dataset("Sigma_not_stored"))
        h5::h5_read(iter_grp, "Sigma_not_stored", sigma_not_stored);
      utils::check(!sigma_not_stored,
                   "read_scf: Sigma_tskij was intentionally omitted from {}/iter{}. "
                   "This slim intermediate iteration cannot be used for restart.",
                   h5_grp, iter);
      // A missing Sigma_tskij without the new flag means Sigma is exactly zero
      // (e.g. an old HF checkpoint).
      if (iter_grp.has_dataset("Sigma_tskij"))
        nda::h5_read(iter_grp, "Sigma_tskij", Sloc);
      else
        Sloc() = 0.0;
    } else if (iter_grp.has_dataset("Heff_skij")) {
      // checkpoint from a qp scf
      auto sys_grp = h5::group(file).open_group("system");
      nda::array<ComplexType, 4> H0(F.shape());
      nda::h5_read(iter_grp, "Heff_skij", Floc);
      nda::h5_read(sys_grp, "H0_skij", H0);
      Floc -= H0;
      Sloc() = 0.0;
    } else {
      utils::check(false, "read_scf: fail to find a scf solution from {}", output+".mbpt.h5");
    }
    h5::h5_read(iter_grp, "mu", mu);
  }
  node_comm.broadcast_n(&iter, 1, 0);
  node_comm.broadcast_n(&mu, 1, 0);
  node_comm.barrier();
  return iter;
}

template<typename shared_array_t>
void read_H0(mpi3::shared_communicator node_comm, std::string output, shared_array_t &H0) {
  if (node_comm.root()) {
    std::string filename = output + ".mbpt.h5";
    h5::file file(filename, 'r');
    auto sys_grp = h5::group(file).open_group("system");

    auto H0_loc = H0.local();
    nda::h5_read(sys_grp, "H0_skij", H0_loc);
  }
  node_comm.barrier();
}

template<typename shared_array_t>
void read_ovlp(mpi3::shared_communicator node_comm, std::string output, shared_array_t &S) {
  if (node_comm.root()) {
    std::string filename = output + ".mbpt.h5";
    h5::file file(filename, 'r');
    auto sys_grp = h5::group(file).open_group("system");

    auto S_loc = S.local();
    nda::h5_read(sys_grp, "S_skij", S_loc);
  }
  node_comm.barrier();
}

template<typename shared_array_t>
void read_dm(mpi3::shared_communicator node_comm, std::string output, long iter, shared_array_t &Dm) {
  if (node_comm.root()) {
    std::string filename = output + ".mbpt.h5";
    h5::file file(filename, 'r');
    auto scf_grp = h5::group(file).open_group("scf");

    if (iter == -1) h5::h5_read(scf_grp, "final_iter", iter);

    utils::check(scf_grp.has_subgroup("iter"+std::to_string(iter)),
                 "read_dm: \"scf/iter{}\" h5 group does not exist.", iter);

    auto Dm_loc = Dm.local();
    auto iter_grp = scf_grp.open_group("iter"+std::to_string(iter));
    nda::h5_read(iter_grp, "Dm_skij", Dm_loc);
  }
  node_comm.barrier();
}

template<typename X_4D_t>
long read_qpscf(mpi3::shared_communicator node_comm,
                X_4D_t &Heff_skij, double &mu, std::string output) {
  long iter;
  if (node_comm.root()) {
    std::string filename = output + ".mbpt.h5";
    h5::file file(filename, 'r');

    auto scf_grp = h5::group(file).open_group("scf");
    h5::h5_read(scf_grp, "final_iter", iter);

    auto iter_grp = scf_grp.open_group("iter"+std::to_string(iter));
    auto Heffloc = Heff_skij.local();
    if (iter_grp.has_dataset("Heff_skij")) {
      // checkpoint from a qp scf
      nda::h5_read(iter_grp, "Heff_skij", Heffloc);
    } else if (iter_grp.has_dataset("F_skij")) {
      // checkpoint from a dyson scf
      nda::h5_read(iter_grp, "F_skij", Heffloc);
      nda::array<ComplexType, 4> H0(Heff_skij.shape());
      auto sys_grp = h5::group(file).open_group("system");
      nda::h5_read(sys_grp, "H0_skij", H0);
      Heffloc += H0;
      if (iter_grp.has_dataset("Sigma_tskij")) {
        app_warning("read_qpscf: Self-energy data is found in {} although qp-scf will omit this term. "
                    "Check if this is what you want!", output+".mbpt.h5");
      }
    } else {
      utils::check(false, "read_qpscf: fail to find a scf solution from {}", output+".mbpt.h5");
    }
    h5::h5_read(iter_grp, "mu", mu);
  }
  node_comm.broadcast_n(&iter, 1, 0);
  node_comm.broadcast_n(&mu, 1, 0);
  node_comm.barrier();
  return iter;
}

template<typename X_4D_t, typename X_3D_t>
void read_qp_MOs(mpi3::shared_communicator node_comm,
                 X_4D_t &sMO_skia, X_3D_t &sE_ska,
                 std::string output, std::string h5_grp, long iter) {
  sMO_skia.win().fence();
  sE_ska.win().fence();
  if (node_comm.root()) {
    std::string filename = output + ".mbpt.h5";
    h5::file file(filename, 'r');
    auto scf_grp = h5::group(file).open_group(h5_grp);
    if (iter == -1) h5::h5_read(scf_grp, "final_iter", iter);
    utils::check(scf_grp.has_subgroup("iter"+std::to_string(iter)),
                 "read_qp_MOs: \"{}/iter{}\" h5 group does not exist in {}.",
                 h5_grp, iter, filename);
    auto iter_grp = scf_grp.open_group("iter"+std::to_string(iter));
    utils::check(iter_grp.has_dataset("MO_skia") && iter_grp.has_dataset("E_ska"),
                 "read_qp_MOs: '{}/iter{}' has no MO_skia/E_ska datasets (not a qp "
                 "checkpoint). Re-run qpGW so the QP eigenbasis is persisted.",
                 h5_grp, iter);
    auto MO_loc = sMO_skia.local();
    auto E_loc = sE_ska.local();
    nda::h5_read(iter_grp, "MO_skia", MO_loc);
    nda::h5_read(iter_grp, "E_ska", E_loc);
  }
  sMO_skia.win().fence();
  sE_ska.win().fence();
  node_comm.barrier();
}

template<typename X_4D_t, typename X_3D_t>
void write_qpgw_results(std::string filename, long gw_iter,
                        const X_3D_t &E_ska,
                        const X_4D_t &MO_skia,
                        const X_4D_t &Vcorr_skij,
                        double mu) {
  app_log(2, "Writing QPGW results to \"scf/iter{}\"\n", gw_iter);
  if (Vcorr_skij.communicator()->root()) {
    auto E_loc = E_ska.local();
    auto MO_loc = MO_skia.local();
    auto Vcorr_loc = Vcorr_skij.local();
    h5::file file(filename, 'a');
    auto iter_grp = h5::group(file).open_group("scf/iter"+std::to_string(gw_iter));
    auto qp_grp = (iter_grp.has_subgroup("qp_approx"))?
                  iter_grp.open_group("qp_approx") : iter_grp.create_group("qp_approx");
    nda::h5_write(qp_grp, "E_ska", E_loc, false);
    nda::h5_write(qp_grp, "MO_skia", MO_loc, false);
    nda::h5_write(qp_grp, "Vcorr_skij", Vcorr_loc, false);
    h5::h5_write(qp_grp, "mu", mu);
  }
  Vcorr_skij.communicator()->barrier();
}

template<typename X_4D_t>
void read_qp_hamilt_components(X_4D_t &Vhf_skij,
                               X_4D_t &Vcorr_skij,
                               double &mu,
                               std::string filename,
                               long gw_iter) {
  h5::file file(filename, 'r');
  auto scf_grp = h5::group(file).open_group("scf/iter" + std::to_string(gw_iter));
  auto qp_grp = scf_grp.open_group("qp_approx");

  h5::read(qp_grp, "mu", mu);

  if (Vcorr_skij.node_comm()->root()) {
    auto Vhf_loc = Vhf_skij.local();
    auto Vcorr_loc = Vcorr_skij.local();

    nda::h5_read(scf_grp, "F_skij", Vhf_loc);
    if (qp_grp.has_dataset("Vcorr_skij"))
      nda::h5_read(qp_grp, "Vcorr_skij", Vcorr_loc);
    else {
      // CNY: backward compatibility... Will be removed in the near future
      nda::h5_read(qp_grp, "Vcorr_skab", Vcorr_loc);
    }
  }
  Vcorr_skij.communicator()->barrier();
}

void write_scf_status(mpi3::communicator &comm, std::string output, std::string status) {
  if (comm.root()) {
    std::string filename = output + ".mbpt.h5";
    h5::file file(filename, 'a');
    h5::group grp(file);
    auto scf_grp = (grp.has_subgroup("scf"))? grp.open_group("scf") : grp.create_group("scf");
    h5::h5_write(scf_grp, "scf_status", status);
  }
  comm.barrier();
}

std::string read_scf_status(std::string filename) {
  h5::file file(filename, 'r');
  h5::group grp(file);
  if (not grp.has_subgroup("scf")) return "unknown";
  auto scf_grp = grp.open_group("scf");
  if (not scf_grp.has_dataset("scf_status")) return "unknown";
  std::string status;
  h5::h5_read(scf_grp, "scf_status", status);
  return status;
}

auto read_input_iterations(std::string filename)
-> std::tuple<long, long, long, long> {
  long gw_iter;
  long weiss_f_iter;
  long weiss_b_iter;
  long embed_iter;

  h5::file file(filename, 'r');

  // gw_iter
  utils::check(h5::group(file).has_subgroup("scf"),
               "embed_t::read_input_iterations: h5 group \"scf\" does not exist in {}", filename);
  auto gw_grp = h5::group(file).open_group("scf");
  h5::h5_read(gw_grp, "final_iter", gw_iter);

  // embed_iter
  if (h5::group(file).has_subgroup("embed")) {
    auto embed_grp = h5::group(file).open_group("embed");
    h5::h5_read(embed_grp, "final_iter", embed_iter);
  } else
    embed_iter = -1;

  // weiss_f_iter
  if (h5::group(file).has_subgroup("downfold_1e")) {
    auto weiss_f_grp = h5::group(file).open_group("downfold_1e");
    h5::h5_read(weiss_f_grp, "final_iter", weiss_f_iter);
  } else
    weiss_f_iter = -1;

  // weiss_b_iter
  if (h5::group(file).has_subgroup("downfold_2e")) {
    auto weiss_b_grp = h5::group(file).open_group("downfold_2e");
    h5::h5_read(weiss_b_grp, "final_iter", weiss_b_iter);
  } else
    weiss_b_iter = -1;

  return std::make_tuple(gw_iter, weiss_f_iter, weiss_b_iter, embed_iter);
}

bool is_qp_selfenergy(std::string filename) {
  long weiss_f_iter;
  h5::file file(filename, 'r');
  auto weiss_f_grp = h5::group(file).open_group("downfold_1e");
  h5::read(weiss_f_grp, "final_iter", weiss_f_iter);
  auto iter_grp = weiss_f_grp.open_group("iter"+std::to_string(weiss_f_iter));
  if (iter_grp.has_dataset("Vcorr_gw_sIab"))
    return true;
  else
    return false;
}

bool read_sigma_local(nda::array<ComplexType, 5> &Sigma_imp_wsIab,
                      nda::array<ComplexType, 4> &Vhf_imp_sIab,
                      std::string filename, long weiss_f_iter) {
  bool sigma_local_exist = false;
  h5::file file(filename, 'r');
  auto root_grp = h5::group(file);
  std::optional<h5::group> weiss_f_grp;
  if (root_grp.has_subgroup("downfold_1e")) {
    auto df_1e_grp = root_grp.open_group("downfold_1e");
    if (df_1e_grp.has_subgroup("iter" + std::to_string(weiss_f_iter)))
      weiss_f_grp = df_1e_grp.open_group("iter" + std::to_string(weiss_f_iter));
  }

  if (weiss_f_grp &&
      weiss_f_grp->has_dataset("Sigma_imp_wsIab") &&
      weiss_f_grp->has_dataset("Vhf_imp_sIab")) {

    nda::h5_read(*weiss_f_grp, "Sigma_imp_wsIab", Sigma_imp_wsIab);
    nda::h5_read(*weiss_f_grp, "Vhf_imp_sIab", Vhf_imp_sIab);
    sigma_local_exist = true;

  }
  return sigma_local_exist;
}

bool read_sigma_local(nda::array<ComplexType, 5> &Sigma_imp_wsIab,
                      nda::array<ComplexType, 4> &Vcorr_dc_sIab,
                      nda::array<ComplexType, 4> &Vhf_imp_sIab,
                      nda::array<ComplexType, 4> &Vhf_dc_sIab,
                      std::string filename, long weiss_f_iter) {
  bool sigma_local_exist = false;
  h5::file file(filename, 'r');
  auto root_grp = h5::group(file);
  std::optional<h5::group> weiss_f_grp;
  if (root_grp.has_subgroup("downfold_1e")) {
    auto df_1e_grp = root_grp.open_group("downfold_1e");
    if (df_1e_grp.has_subgroup("iter" + std::to_string(weiss_f_iter)))
      weiss_f_grp = df_1e_grp.open_group("iter" + std::to_string(weiss_f_iter));
  }

  if (weiss_f_grp &&
      weiss_f_grp->has_dataset("Sigma_imp_wsIab") &&
      weiss_f_grp->has_dataset("Vcorr_dc_sIab") &&
      weiss_f_grp->has_dataset("Vhf_imp_sIab") &&
      weiss_f_grp->has_dataset("Vhf_dc_sIab")) {

    nda::h5_read(*weiss_f_grp, "Sigma_imp_wsIab", Sigma_imp_wsIab);
    nda::h5_read(*weiss_f_grp, "Vcorr_dc_sIab", Vcorr_dc_sIab);
    nda::h5_read(*weiss_f_grp, "Vhf_imp_sIab", Vhf_imp_sIab);
    nda::h5_read(*weiss_f_grp, "Vhf_dc_sIab", Vhf_dc_sIab);
    sigma_local_exist = true;
  }
  return sigma_local_exist;
}

bool read_sigma_local(nda::array<ComplexType, 5> &Sigma_imp_wsIab,
                      nda::array<ComplexType, 5> &Sigma_dc_wsIab,
                      nda::array<ComplexType, 4> &Vhf_imp_sIab,
                      nda::array<ComplexType, 4> &Vhf_dc_sIab,
                      std::string filename, long weiss_f_iter) {
  bool sigma_local_exist = false;
  h5::file file(filename, 'r');
  auto root_grp = h5::group(file);
  std::optional<h5::group> weiss_f_grp;
  if (root_grp.has_subgroup("downfold_1e")) {
    auto df_1e_grp = root_grp.open_group("downfold_1e");
    if (df_1e_grp.has_subgroup("iter" + std::to_string(weiss_f_iter)))
      weiss_f_grp = df_1e_grp.open_group("iter" + std::to_string(weiss_f_iter));
  }

  if (weiss_f_grp &&
      weiss_f_grp->has_dataset("Sigma_imp_wsIab") &&
      weiss_f_grp->has_dataset("Sigma_dc_wsIab") &&
      weiss_f_grp->has_dataset("Vhf_imp_sIab") &&
      weiss_f_grp->has_dataset("Vhf_dc_sIab")) {

    nda::h5_read(*weiss_f_grp, "Sigma_imp_wsIab", Sigma_imp_wsIab);
    nda::h5_read(*weiss_f_grp, "Sigma_dc_wsIab", Sigma_dc_wsIab);
    nda::h5_read(*weiss_f_grp, "Vhf_imp_sIab", Vhf_imp_sIab);
    nda::h5_read(*weiss_f_grp, "Vhf_dc_sIab", Vhf_dc_sIab);
    sigma_local_exist = true;
  }
  return sigma_local_exist;
}

template<typename shared_array_t>
bool read_pi_local(shared_array_t &sPi_imp, shared_array_t &sPi_dc,
                   std::string filename, long weiss_b_iter) {
  bool pi_local_exist = false;
  auto Pi_imp = sPi_imp.local();
  auto Pi_dc = sPi_dc.local();
  if (sPi_imp.node_comm()->root()) {
    h5::file file(filename, 'r');
    auto root_grp = h5::group(file);

    std::optional<h5::group> weiss_b_grp;
    if (root_grp.has_subgroup("downfold_2e")) {
      auto df_2e_grp = root_grp.open_group("downfold_2e");

      if (weiss_b_iter == -1)
        h5::h5_read(df_2e_grp, "final_iter", weiss_b_iter);

      if (df_2e_grp.has_subgroup("iter" + std::to_string(weiss_b_iter)))
        weiss_b_grp = df_2e_grp.open_group("iter" + std::to_string(weiss_b_iter));
    }

    if (weiss_b_grp && weiss_b_grp->has_dataset("Pi_imp_wabcd")
        && weiss_b_grp->has_dataset("Pi_dc_wabcd")) {
      nda::h5_read(*weiss_b_grp, "Pi_imp_wabcd", Pi_imp);
      nda::h5_read(*weiss_b_grp, "Pi_dc_wabcd", Pi_dc);
      pi_local_exist = true;
    }
  }
  sPi_imp.node_comm()->broadcast_n(&pi_local_exist, 1, 0);
  sPi_imp.communicator()->barrier();

  return pi_local_exist;
}




/** Public template instantiation **/

template void write_metadata(
    mpi3::communicator&, const mf::MF&, const imag_axes_ft::IAFT&,
    const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_4D_t>&,
    std::string);

template void dump_scf(
    mpi3::communicator&, long,
    const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_5D_t>&,
    const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_5D_t>&,
    double, std::string, std::string, long, bool, const sArray_t<Array_view_4D_t>*);

template void dump_scf(
    mpi3::communicator&, long,
    const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_3D_t>&,
    double, std::string);

template long read_scf(
    mpi3::shared_communicator, sArray_t<Array_view_4D_t>&,
    sArray_t<Array_view_5D_t>&, double&, std::string, std::string, long);

template void read_H0(mpi3::shared_communicator, std::string, sArray_t<Array_view_4D_t>&);
template void read_ovlp(mpi3::shared_communicator, std::string, sArray_t<Array_view_4D_t>&);
template void read_dm(mpi3::shared_communicator, std::string, long, sArray_t<Array_view_4D_t>&);

template long read_qpscf(
    mpi3::shared_communicator, sArray_t<Array_view_4D_t>&,
    double&, std::string);

template void write_qpgw_results(
    std::string, long, const sArray_t<Array_view_3D_t>&,
    const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_4D_t>&, double);

template void read_qp_hamilt_components(
    sArray_t<Array_view_4D_t>&, sArray_t<Array_view_4D_t>&,
    double &, std::string, long);

template bool read_pi_local(sArray_t<Array_view_5D_t>&, sArray_t<Array_view_5D_t>&, std::string, long);

template<typename communicator_t, typename shared_array_t>
bool read_DeltaH0(communicator_t& comm,
                  std::string filename,
                  nda::array<double, 1>& q_vec,
                  shared_array_t& sDeltaH0_skij) {
  bool success = false;
  if (comm.root()) {
    try {
      h5::file file(filename, 'r');
      auto root_grp = h5::group(file);

      if (root_grp.has_subgroup("linear_response")) {
        auto lr_grp = root_grp.open_group("linear_response");

        if (lr_grp.has_dataset("q_vec") && lr_grp.has_dataset("DeltaH0_skij")) {
          nda::h5_read(lr_grp, "q_vec", q_vec);
          auto DeltaH0_loc = sDeltaH0_skij.local();
          nda::h5_read(lr_grp, "DeltaH0_skij", DeltaH0_loc);
          success = true;
        }
      }
    } catch (const std::exception& e) {
      app_warning("read_DeltaH0: Failed to read from {}: {}", filename, e.what());
      success = false;
    }
  }
  comm.broadcast_n(&success, 1, 0);
  if (success) {
    if (!comm.root()) q_vec.resize(3);
    comm.broadcast_n(q_vec.data(), 3, 0);
  }
  comm.barrier();
  return success;
}

template<typename communicator_t, typename shared_array_t>
void write_DeltaH0(communicator_t& comm,
                   std::string filename,
                   nda::array<double, 1> const& q_vec,
                   shared_array_t const& sDeltaH0_skij) {
  if (comm.root()) {
    utils::check(std::filesystem::exists(filename),
                 "write_DeltaH0: File {} does not exist. Cannot append.", filename);
    h5::file file(filename, 'a');
    h5::group grp(file);
    auto lr_grp = grp.has_subgroup("linear_response") ?
                  grp.open_group("linear_response") :
                  grp.create_group("linear_response");

    nda::h5_write(lr_grp, "q_vec", q_vec, false);
    auto DeltaH0_loc = sDeltaH0_skij.local();
    nda::h5_write(lr_grp, "DeltaH0_skij", DeltaH0_loc, false);
  }
  comm.barrier();
}

template<typename communicator_t, typename G_t, typename Dm_t, typename F_t, typename Sigma_t>
void dump_lr(communicator_t& comm,
             std::string filename,
             nda::array<double, 1> const& q_vec,
             G_t const& sDeltaG_tskij,
             Dm_t const& sDeltaDm_skij,
             F_t const& sDeltaF_skij,
             Sigma_t const* sDeltaSigma_tskij,
             double Delta_mu,
             int niter,
             bool include_hartree,
             bool include_exchange,
             bool include_gw_sigma,
             Sigma_t const* sDeltaSigma2_tskij,
             F_t const* sDeltaVcorr_skij,
             std::optional<long> imode,
             bool save_DeltaG,
             std::optional<long> nbnd_save,
             std::string const& gw_mode,
             bool lr_two_step,
             std::string const& two_step_inner_method,
             int two_step_order,
             bool two_step_outer_accel,
             std::string const& two_step_outer_alg,
             double two_step_outer_tol,
             int two_step_stages_applied,
             std::optional<double> exchange_static_W_head) {
  if (comm.root()) {
    utils::check(std::filesystem::exists(filename),
                 "dump_lr: File {} does not exist. Cannot append.", filename);
    h5::file file(filename, 'a');
    h5::group grp(file);
    auto lr_grp = grp.has_subgroup("linear_response") ?
                  grp.open_group("linear_response") :
                  grp.create_group("linear_response");
    // A multi-perturbation run gives each one its own subgroup; a single one
    // keeps writing straight into "linear_response/", as before.
    if (imode) {
      std::string m = "mode" + std::to_string(*imode);
      lr_grp = lr_grp.has_subgroup(m) ? lr_grp.open_group(m) : lr_grp.create_group(m);
    }

    // Write an imaginary-time array, trimmed to the leading nbnd_save x nbnd_save
    // band block when asked. A trimmed dataset is the protected-band block of the
    // quantity, not the full-basis object, so it is stamped with the band count
    // it was cut to; untrimmed datasets carry no attribute.
    auto write_tskij = [&](std::string const& name, auto const& A) {
      if (nbnd_save) {
        long nb = A.shape(3);
        utils::check(*nbnd_save >= 0 and *nbnd_save <= nb,
                     "dump_lr: nbnd_save must be in [0, {}], got {}", nb, *nbnd_save);
        auto rng = nda::range(0, *nbnd_save);
        nda::h5_write(lr_grp, name,
                      nda::make_regular(A(nda::ellipsis{}, rng, rng)), false);
        h5::h5_write_attribute(lr_grp.open_dataset(name), "nbnd_save", *nbnd_save);
      } else {
        nda::h5_write(lr_grp, name, A, false);
      }
    };

    nda::h5_write(lr_grp, "q_vec", q_vec, false);
    auto DeltaDm_loc = sDeltaDm_skij.local();
    if (save_DeltaG) write_tskij("DeltaG_tskij", sDeltaG_tskij.local());
    nda::h5_write(lr_grp, "DeltaDm_skij", DeltaDm_loc, false);
    h5::h5_write(lr_grp, "Delta_mu", Delta_mu);
    h5::h5_write(lr_grp, "niter", niter);

    h5::h5_write(lr_grp, "include_hartree", static_cast<int>(include_hartree));
    h5::h5_write(lr_grp, "include_exchange", static_cast<int>(include_exchange));
    h5::h5_write(lr_grp, "include_gw_sigma", static_cast<int>(include_gw_sigma));
    // Refines include_gw_sigma, which cannot distinguish fixed_W from full.
    if (!gw_mode.empty()) h5::h5_write(lr_grp, "gw_mode", gw_mode);
    // HSEX: include_exchange alone cannot say which interaction ΔK contracted.
    // Additive, so a bare-exchange checkpoint keeps exactly the fields it had.
    if (exchange_static_W_head) {
      h5::h5_write(lr_grp, "exchange_static_W", 1);
      h5::h5_write(lr_grp, "exchange_static_W_head", *exchange_static_W_head);
    }

    if (include_hartree || include_exchange) {
      auto DeltaF_loc = sDeltaF_skij.local();
      nda::h5_write(lr_grp, "DeltaF_skij", DeltaF_loc, false);
    }
    if (include_gw_sigma) {
      utils::check(sDeltaSigma_tskij != nullptr,
                   "dump_lr: include_gw_sigma=true but sDeltaSigma_tskij is null.");
      write_tskij("DeltaSigma_tskij", sDeltaSigma_tskij->local());
    }
    // Split-term one-shot G0W0 output (opt-in): DeltaSigma_tskij above holds the
    // TOTAL correlation ΔΣ = dG0·W_c0 + G0·dW0 (same as the fused output); this
    // breaks out the G0·dW0 piece. So dG0·W_c0 = DeltaSigma_tskij - DeltaSigma_GdW,
    // and the full dG0·W0 = DeltaF (exchange) + (DeltaSigma_tskij - DeltaSigma_GdW).
    // The flag + GdW dataset are written only in split mode, so the standard
    // (fused) output format is unchanged.
    if (sDeltaSigma2_tskij != nullptr) {
      h5::h5_write(lr_grp, "split_sigma_terms", 1);
      write_tskij("DeltaSigma_GdW_tskij", sDeltaSigma2_tskij->local());
    }
    // Static ΔV_QPGW (LR-qpGW): the frequency-independent correlation potential
    // response that entered the Dyson RHS in place of the dynamic ΔΣ. Written
    // only in qp mode (additive dataset; standard output format is unchanged).
    if (sDeltaVcorr_skij != nullptr) {
      h5::h5_write(lr_grp, "qp_static_sigma", 1);
      auto DeltaVcorr_loc = sDeltaVcorr_skij->local();
      nda::h5_write(lr_grp, "DeltaVcorr_skij", DeltaVcorr_loc, false);
    }
    // Split-kernel (two-step) schedule. DeltaF_skij / DeltaSigma_tskij above are
    // then the sums of the two channels, with the perturbative part evaluated at
    // the ΔG of the previous stage — so neither is a self-consistent response to
    // the ΔG that was written.
    if (lr_two_step) {
      h5::h5_write(lr_grp, "lr_two_step", 1);
      h5::h5_write(lr_grp, "two_step_inner_method", two_step_inner_method);
      h5::h5_write(lr_grp, "two_step_order", two_step_order);
      // Outer-loop acceleration. Written only when it is actually active, so a
      // plain two-step checkpoint keeps exactly the fields it had before. With
      // acceleration on, two_step_order is an iteration cap rather than a
      // truncation order and the result carries no order interpretation, so
      // two_step_stages_applied — the number of K_pert evaluations actually
      // made — is the only honest cost/provenance record.
      if (two_step_outer_accel) {
        h5::h5_write(lr_grp, "two_step_outer_alg", two_step_outer_alg);
        h5::h5_write(lr_grp, "two_step_outer_tol", two_step_outer_tol);
        h5::h5_write(lr_grp, "two_step_stages_applied", two_step_stages_applied);
      }
    }

    app_log(2, "LR results written to \"{}\" in {}",
            imode ? fmt::format("linear_response/mode{}/", *imode) : "linear_response/",
            filename);
    app_log(2, "  - niter = {}, Delta_mu = {:.6e}", niter, Delta_mu);
  }
  comm.barrier();
}

template<typename communicator_t>
void dump_qp_params(communicator_t& comm, std::string filename,
                    std::string const& off_diag_mode, double eta,
                    std::string const& ac_alg, int Nfit,
                    std::string const& div_treatment) {
  if (comm.root()) {
    utils::check(std::filesystem::exists(filename),
                 "dump_qp_params: File {} does not exist. Cannot append.", filename);
    h5::file file(filename, 'a');
    h5::group grp(file);
    utils::check(grp.has_subgroup("scf"),
                 "dump_qp_params: '{}' has no 'scf' group.", filename);
    auto scf_grp = grp.open_group("scf");
    auto qp_grp = scf_grp.has_subgroup("qp_params") ?
                  scf_grp.open_group("qp_params") : scf_grp.create_group("qp_params");
    h5::h5_write(qp_grp, "off_diag_mode", off_diag_mode);
    h5::h5_write(qp_grp, "eta", eta);
    h5::h5_write(qp_grp, "ac_alg", ac_alg);
    h5::h5_write(qp_grp, "Nfit", Nfit);
    // Stash div_treatment on the scf group (same convention as
    // scr_coulomb_t::dump_eps_inv_head) so LR reconstructs the head consistently.
    if (!scf_grp.has_dataset("div_treatment"))
      h5::h5_write(scf_grp, "div_treatment", div_treatment);
  }
  comm.barrier();
}

template<typename communicator_t>
void dump_hf_div_treatment(communicator_t& comm, std::string filename,
                           std::string const& hf_div_treatment) {
  if (comm.root()) {
    utils::check(std::filesystem::exists(filename),
                 "dump_hf_div_treatment: File {} does not exist. Cannot append.", filename);
    h5::file file(filename, 'a');
    h5::group grp(file);
    // Same create-if-absent convention as scr_coulomb_t::dump_eps_inv_head: this
    // may run before the first dump_scf on a restart.
    auto scf_grp = grp.has_subgroup("scf") ? grp.open_group("scf") : grp.create_group("scf");
    if (!scf_grp.has_dataset("hf_div_treatment"))
      h5::h5_write(scf_grp, "hf_div_treatment", hf_div_treatment);
  }
  comm.barrier();
}

template<typename communicator_t>
bool read_qp_params(communicator_t& comm, std::string filename,
                    std::string& off_diag_mode, double& eta,
                    std::string& ac_alg, int& Nfit) {
  int found = 0;
  char odm_buf[64] = {0};
  char ac_buf[64] = {0};
  if (comm.root()) {
    h5::file file(filename, 'r');
    auto grp = h5::group(file);
    if (grp.has_subgroup("scf")) {
      auto scf_grp = grp.open_group("scf");
      if (scf_grp.has_subgroup("qp_params")) {
        auto qp_grp = scf_grp.open_group("qp_params");
        h5::h5_read(qp_grp, "off_diag_mode", off_diag_mode);
        h5::h5_read(qp_grp, "eta", eta);
        h5::h5_read(qp_grp, "ac_alg", ac_alg);
        h5::h5_read(qp_grp, "Nfit", Nfit);
        utils::check(off_diag_mode.size() < sizeof(odm_buf) && ac_alg.size() < sizeof(ac_buf),
                     "read_qp_params: string field too long.");
        std::copy(off_diag_mode.begin(), off_diag_mode.end(), odm_buf);
        std::copy(ac_alg.begin(), ac_alg.end(), ac_buf);
        found = 1;
      }
    }
  }
  comm.broadcast_n(&found, 1, 0);
  if (found) {
    comm.broadcast_n(&eta, 1, 0);
    comm.broadcast_n(&Nfit, 1, 0);
    comm.broadcast_n(odm_buf, sizeof(odm_buf), 0);
    comm.broadcast_n(ac_buf, sizeof(ac_buf), 0);
    off_diag_mode = std::string(odm_buf);
    ac_alg = std::string(ac_buf);
  }
  comm.barrier();
  return found != 0;
}

// LR template instantiations
template bool read_DeltaH0(mpi3::shared_communicator&, std::string,
                           nda::array<double, 1>&, sArray_t<Array_view_4D_t>&);

template void dump_qp_params(mpi3::communicator&, std::string,
                             std::string const&, double, std::string const&, int,
                             std::string const&);
template bool read_qp_params(mpi3::communicator&, std::string,
                             std::string&, double&, std::string&, int&);
template void dump_hf_div_treatment(mpi3::communicator&, std::string, std::string const&);

template void read_qp_MOs(mpi3::shared_communicator,
                          sArray_t<Array_view_4D_t>&, sArray_t<Array_view_3D_t>&,
                          std::string, std::string, long);

template void write_DeltaH0(mpi3::communicator&, std::string,
                            nda::array<double, 1> const&,
                            sArray_t<Array_view_4D_t> const&);

template void dump_lr(mpi3::communicator&, std::string,
                      nda::array<double, 1> const&,
                      sArray_t<Array_view_5D_t> const&,
                      sArray_t<Array_view_4D_t> const&,
                      sArray_t<Array_view_4D_t> const&,
                      sArray_t<Array_view_5D_t> const*,
                      double, int, bool, bool, bool,
                      sArray_t<Array_view_5D_t> const*,
                      sArray_t<Array_view_4D_t> const*,
                      std::optional<long>, bool, std::optional<long>,
                      std::string const&,
                      bool, std::string const&, int,
                      bool, std::string const&, double, int,
                      std::optional<double>);

  } // chkpt
} // methods
