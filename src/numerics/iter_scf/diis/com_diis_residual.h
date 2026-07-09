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


#ifndef COQUI_COM_DIIS_RESIDUAL_H
#define COQUI_COM_DIIS_RESIDUAL_H

#include "numerics/iter_scf/diis/vspace.h"
#include "numerics/iter_scf/diis/vspace_fock_sigma.hpp"
#include "numerics/iter_scf/diis/diis_residual.h"

namespace iter_scf {


class com_diis_residual : public diis_residual<FockSigma> {
    using Array_4D = nda::array<ComplexType,4>;
    using Array_5D = nda::array<ComplexType,5>;
protected:
    using diis_residual<FockSigma>::is_initialized;

    bool com_initialized = false; // full initialization flag

    const imag_axes_ft::IAFT *FT = nullptr;
    Array_4D _S;  // Overlap matrix
    Array_4D _H0; // Non-interacting Hamiltonian
    double mu;    // Chemical potential

    Array_5D G_incoming;
    long iter = -1;
    std::string mbpt_output;
    bool g_mu_injected = false;

    Array_5D C_t_injected;         // pre-computed commutator (A10 distributed path)
    bool residual_injected = false;

public:

    /**
     * Supply a pre-computed commutator residual C_t (A10 k-striped distributed
     * path). The next get_diis_residual consumes it once instead of running the
     * serial commutator_t; if not injected, get_diis_residual falls back to the
     * serial path.
     */
    template<nda::MemoryArrayOfRank<5> Array_C>
    void upload_residual(const Array_C& C_t_) {
        diis_timers::res_inject_copy.start();
        C_t_injected = C_t_;
        diis_timers::res_inject_copy.stop();
        residual_injected = true;
    }

    // a version with external G
    void upload_g(Array_5D& G_) {
        G_incoming = G_;
    }
    /**
     * Supply the latest-iteration G and mu from memory (byte-identical to the
     * scf/iter{final_iter} checkpoint datasets), so the next get_diis_residual
     * skips the multi-GB re-read of G_tskij from the checkpoint file.
     */
    template<nda::MemoryArrayOfRank<5> Array_G>
    void upload_g_mu(const Array_G& G_, double mu_) {
        diis_timers::gmu_inject_copy.start();
        G_incoming = G_;
        diis_timers::gmu_inject_copy.stop();
        mu = mu_;
        g_mu_injected = true;
    }
    // a version with external mu
    void update_mu(double mu_) {
        mu = mu_;
    }
    // read G and mu from file
    void upload_g_mu() {
        long iter_from_file;
        std::string filename = mbpt_output + ".mbpt.h5";
        h5::file file(filename, 'r');
        h5::group grp(file);
        utils::check(grp.has_subgroup("scf"), "Simulation HDF5 file does not have an scf group");
        auto scf_grp = grp.open_group("scf");
        h5::h5_read(scf_grp, "final_iter", iter_from_file);
        if(iter != iter_from_file) {
            auto iter_grp = scf_grp.open_group("iter"+std::to_string(iter_from_file));
            h5::h5_read(iter_grp, "mu", mu);
            utils::check(iter_grp.has_dataset("G_tskij"),
                "com_diis_residual: scf/iter{} has no G_tskij (chkpt_slim intermediate "
                "iteration); this fallback path requires in-memory G injection "
                "(upload_g_mu(G, mu)) or a full checkpoint.", iter_from_file);
            nda::h5_read(iter_grp, "G_tskij", G_incoming);
            iter = iter_from_file;
        }
    }
    
    // These are very limited constructors to use from a very limited interface....
    // Do not lead to full initialization
    com_diis_residual() = default;
    com_diis_residual(VSpace<FockSigma>* x_space) {
        x_vsp = x_space;
    }

    /**
     * Initialization of the commutator residual
     * @param state_         - [INPUT] Pointer to the current state
     * @param S              - [INPUT] Overlap matrix
     * @param H0             - [INPUT] Non-interacting Hamiltonian
     * @param FT_            - [INPUT] Imaginary frequency FT axes
     * @param mbpt_output_   - [INPUT] HDF5 output prefix for reading G and mu
     */
    void initialize(opt_state<FockSigma> *current_state_, const Array_4D& S, const Array_4D& H0,
                    const imag_axes_ft::IAFT *FT_, std::string mbpt_output_) {
        if(!com_initialized) {
            current_state = current_state_;
            FT = FT_;
            _S = S;
            _H0 = H0;
            mbpt_output= mbpt_output_;
        }
    
        is_initialized = true;
        com_initialized = true;
    }



    // Commutator residual
    // This may not be the most memory-efficient implementation...
    bool get_diis_residual(FockSigma& res) override {
        utils::check(com_initialized, "DIIS commutator residual is not initialized");
            // Warning! Sigma here is in tau!
            const FockSigma& x_last = current_state->get_ref();

            // The residual Fock block is zero; a fresh complex nda array is
            // calloc-zeroed (nda::mem::init_dcmplx), so no copy or explicit fill.
            diis_timers::res_fz.start();
            Array_4D Fz(x_last.get_fock().shape());
            diis_timers::res_fz.stop();

            if (residual_injected) {
                // A10: consume the pre-computed distributed commutator, moving it
                // into res (the single unavoidable copy already happened in
                // upload_residual, out of the node-shared window).
                residual_injected = false;
                diis_timers::res_set_copy.start();
                res.set_fock_sigma(Fz, std::move(C_t_injected));
                diis_timers::res_set_copy.stop();
                return true;
            }

            if (g_mu_injected) g_mu_injected = false; // consume the injected G/mu
            else upload_g_mu();

            Array_5D C_t;
            commutator_t(C_t, FT, G_incoming, x_last, mu, _S, _H0);
            diis_timers::res_set_copy.start();
            res.set_fock_sigma(Fz, std::move(C_t));
            diis_timers::res_set_copy.stop();

            return true;
        }
    };


}


#endif // COQUI_COM_DIIS_RESIDUAL_H
