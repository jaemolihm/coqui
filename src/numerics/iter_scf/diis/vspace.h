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


#ifndef COQUI_VECTOR_SPACE_H
#define COQUI_VECTOR_SPACE_H

#include <vector>
#include <complex>
#include <type_traits>

#include "numerics/iter_scf/diis/diis_timers.hpp"

namespace iter_scf {

// Abstract vector spaces and operations used in DIIS
// Vectors are stored in an external HDF5 file (default) or, opt-in, in memory
// (storage = "memory"): the file round-trip per overlap/get/purge dominates the
// DIIS cost for large Fock/Sigma vectors, at the price of holding
// max_subsp_size vectors in RAM on the calling rank.
// The class must be initialized before usage

template<typename Vector>
class VSpace {
private:
    size_t _size; // Subspace size
    std::string _filename; // name of the file containing the vector space
    bool _in_memory = false;
    std::vector<Vector> _vecs; // vector storage when in-memory
    bool inited = false;

    // Detect whether Vector exposes the H7 conjugate-handle interface
    // (make_conj_flat + dot_prod_conj_lhs). FockSigma does; Heff does not.
    template<typename V>
    static auto detect_conj_handle(int)
        -> decltype(std::declval<const V&>().make_conj_flat(), std::true_type{});
    template<typename V>
    static std::false_type detect_conj_handle(...);
    static constexpr bool has_conj_handle =
        decltype(detect_conj_handle<Vector>(0))::value;

public:

    VSpace() {
        _size = 0;
    }

    VSpace(std::string filename) : _filename(filename) {
        _size = 0;
        inited = true;
    }

    void initialize(std::string filename, bool in_memory = false) {
        if(!inited) {
            _filename = filename;
            _in_memory = in_memory;
            inited = true;
        }
    }

    Vector get_vec(const size_t i) {
        Vector vec;
        get_vec(i, vec);
        return vec;
    };

    /**
     * Get vector from the vector space (in-memory copy or read from the h5 file)
     */
    void get_vec(const size_t i, Vector& vec) {
        utils::check(inited, "VSpace is not initialized");
        utils::check(i < _size, "VSpace::get_vec Vector index of the VSpace container {} is out of bounds", _filename);
        diis_timers::vsp_get.start();
        if (_in_memory) vec = _vecs[i];
        else vec.read_from_file(_filename, i);
        diis_timers::vsp_get.stop();
    }

    /**
     * Add vector to the vector space (in-memory copy or write to the h5 file)
     */
    void add_to_vspace(const Vector& a) {
        utils::check(inited, "VSpace is not initialized");
        diis_timers::vsp_add.start();
        if (_in_memory) _vecs.push_back(a);
        else a.write_to_file(_filename, _size);
        diis_timers::vsp_add.stop();
        _size++;
    }

    // Move overload: in-memory storage takes ownership without a copy; the disk
    // path writes and then drops the vector, exactly as the const& overload.
    void add_to_vspace(Vector&& a) {
        utils::check(inited, "VSpace is not initialized");
        diis_timers::vsp_add.start();
        if (_in_memory) _vecs.push_back(std::move(a));
        else a.write_to_file(_filename, _size);
        diis_timers::vsp_add.stop();
        _size++;
    }

    std::complex<double> overlap(const size_t i, const size_t j) {
        utils::check(inited, "VSpace is not initialized");
        utils::check(i < _size, "VSpace::overlap Vector index of the VSpace container {} is out of bounds", _filename);
        utils::check(j < _size, "VSpace::overlap Vector index of the VSpace container {} is out of bounds", _filename);
        if (_in_memory) return overlap(_vecs[i], _vecs[j]);
        Vector vec_i;
        vec_i.read_from_file(_filename, i);
        Vector vec_j;
        vec_j.read_from_file(_filename, j);
        return overlap(vec_i, vec_j);
    }

    std::complex<double> overlap(const size_t i, const Vector& a) {
        utils::check(inited, "VSpace is not initialized");
        utils::check(i < _size, "VSpace::overlap Vector index of the VSpace container {} is out of bounds", _filename);
        if (_in_memory) return overlap(_vecs[i], a);
        Vector vec_i;
        vec_i.read_from_file(_filename, i);
        return overlap(vec_i, a);
    }

    std::complex<double> overlap(const Vector& a, const Vector& b) {
        // Leaf overlap: index-based overlap(i,j)/(i,a) all funnel here, so timing
        // this alone captures every dot_prod without Watch re-entrancy.
        diis_timers::vsp_overlap.start();
        auto res = a.dot_prod(b);
        diis_timers::vsp_overlap.stop();
        return res;
    }

    // H7: build the new DIIS B-matrix row/column against the incoming residual u.
    // Returns { <x_0|u>, ..., <x_{n-1}|u>, <u|u> } with n = current subspace size.
    // For a Vector exposing a conjugate handle (FockSigma) this materializes conj(u)
    // once and reuses it across every x_i (and u), computing <x_i|u> as
    // conj(<u|x_i>) — bit-identical to overlap(i,u)/overlap(u,u) because the gemm
    // kernel/shapes are unchanged and only the conjugated operand is swapped. Vector
    // types without the handle (Heff) fall back to the unchanged per-pair path.
    std::vector<std::complex<double>> overlaps_new_row(const Vector& u) {
        utils::check(inited, "VSpace is not initialized");
        std::vector<std::complex<double>> out(_size + 1);
        // COQUI_DEBUG_PAIR_OVERLAP=1 forces the pre-H7 per-pair path (same binary
        // A/B for the digit-identity gate).
        static const bool force_pair = []() {
            const char* e = std::getenv("COQUI_DEBUG_PAIR_OVERLAP");
            return e && e[0] == '1';
        }();
        if constexpr (has_conj_handle) {
          if (!force_pair) {
            diis_timers::vsp_overlap.start();
            auto u_conj = u.make_conj_flat();
            diis_timers::vsp_overlap.stop();
            Vector scratch; // disk-mode read target (untimed, as in overlap(i,u))
            for (size_t i = 0; i < _size; i++) {
                const Vector* xi;
                if (_in_memory) { xi = &_vecs[i]; }
                else { scratch.read_from_file(_filename, i); xi = &scratch; }
                diis_timers::vsp_overlap.start();
                out[i] = std::conj(xi->dot_prod_conj_lhs(u_conj)); // <x_i|u>
                diis_timers::vsp_overlap.stop();
            }
            diis_timers::vsp_overlap.start();
            out[_size] = u.dot_prod_conj_lhs(u_conj); // <u|u> (matches overlap(u,u))
            diis_timers::vsp_overlap.stop();
            return out;
          }
        }
        for (size_t i = 0; i < _size; i++) out[i] = overlap(i, u);
        out[_size] = overlap(u, u);
        return out;
    }

    size_t size() {
        return _size; 
    };

    // TODO: implement through move
    void purge_vec(const size_t k) {
        utils::check(inited, "VSpace is not initialized");
        utils::check(k < _size, "VSpace::purge_vec Vector index of the VSpace container {} is out of bounds", _filename);
        utils::check(_size > 0, "VSpace::purge_vec VSpace is of zero size, no vector can be deleted");
        if (_in_memory) {
            _vecs.erase(_vecs.begin() + k);
        } else {
            Vector vec;
            for(size_t j = k+1; j < size(); j++) {
                vec.read_from_file(_filename, j);
                vec.write_to_file(_filename, j-1);
            }
        }

        _size--;
    }

    virtual Vector make_linear_comb(const nda::array<ComplexType, 1>& C) {
        utils::check(inited, "VSpace is not initialized");
         diis_timers::vsp_lincomb.start();
         Vector r;
         if(_size > 0) {
             get_vec(size()-1, r); // this is needed to initialize r
             r.set_zero();
         }
         else { diis_timers::vsp_lincomb.stop(); return r; }
         for(size_t i = 0; i < _size && i < C.size(); i++) {
             ComplexType coeff = C(C.size()-1-i);
             // In-memory: accumulate directly from the stored vector (no copy).
             // Disk: read the vector into a temporary and accumulate.
             if (_in_memory) r.add(_vecs[size()-1-i], coeff);
             else r.add(get_vec(size()-1-i), coeff);
         }
         diis_timers::vsp_lincomb.stop();
         return r;
     }

};


} // namespace
#endif //  COQUI_VECTOR_SPACE_H
