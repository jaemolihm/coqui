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
        if (_in_memory) vec = _vecs[i];
        else vec.read_from_file(_filename, i);
    }

    /**
     * Add vector to the vector space (in-memory copy or write to the h5 file)
     */
    void add_to_vspace(const Vector& a) {
        utils::check(inited, "VSpace is not initialized");
        if (_in_memory) _vecs.push_back(a);
        else a.write_to_file(_filename, _size);
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
        return a.dot_prod(b);
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
         Vector r;
         if(_size > 0) {
             get_vec(size()-1, r); // this is needed to initialize r
             r.set_zero();
         }
         else return r; 
         for(size_t i = 0; i < _size && i < C.size(); i++) {
             ComplexType coeff = C(C.size()-1-i);
             r.add(get_vec(size()-1-i), coeff);
         }
         return r;
     }

};


} // namespace
#endif //  COQUI_VECTOR_SPACE_H
