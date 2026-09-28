// -*- coding: utf-8 -*-
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <list>
#include <utility>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <omp.h>

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif
#include <cblas.h>

using namespace pybind11::literals;
namespace py = pybind11;

constexpr double TIE_TOL = 1e-5;
using index_t = std::int64_t;
using count_t = std::int64_t;

inline bool is_tied(double a, double b) {
    return std::abs(a - b) <= TIE_TOL;
}

struct Entry {
    index_t data_index; // original index
    index_t group_id;   // group id for cell
    double value;   // original value

    // sorted by group and value
    bool operator<(const Entry& other) const {
        if (group_id != other.group_id) {
            return group_id < other.group_id;
        }
        return value < other.value;
    }
};

struct DenseEntry {
    index_t row_idx;   // row index
    index_t group_id;  // group id for cell
    double value;  // original value

    bool operator<(const DenseEntry& other) const {
        if (group_id != other.group_id) {
            return group_id < other.group_id;
        }
        return value < other.value;
    }
};

// Sum groups for csc matrix (compressed sparse column)
py::array_t<double> cpp_sumGroups_csc(
    py::array_t<double> x,         // non-zero element values (by column)
    py::array_t<index_t> p,        // The start and end indices of each column's non-zero elements
    py::array_t<index_t> i,        // The row index corresponding to each non-zero element
    index_t ncol,                  // column number of the matrix
    py::array_t<index_t> groups,   // The group index for each row
    index_t ngroups,               // Total number of groups
    int nthreads                   // number of threads to use
) {
    // get read-only access to input data
    auto x_ = x.unchecked<1>();
    auto p_ = p.unchecked<1>();
    auto i_ = i.unchecked<1>();
    auto groups_ = groups.unchecked<1>();

    // creat result matrix
    py::array_t<double, py::array::f_style> res({ngroups, ncol});
    auto res_ = res.mutable_unchecked<2>();

    // initialize res_ to 0
    std::memset(res_.mutable_data(0, 0), 0, sizeof(double) * ngroups * ncol);

    // Set number of threads
    const int max_threads = omp_get_max_threads();
    const int threads_to_use = std::clamp(
        nthreads > 0 ? nthreads : max_threads,
        1,
        max_threads
    );

    {
        py::gil_scoped_release release;

        #pragma omp parallel for num_threads(threads_to_use) schedule(dynamic)
        for (index_t c = 0; c < ncol; ++c) {
            for (index_t j = p_(c); j < p_(c + 1); ++j) {
                index_t row = i_(j);
                if (row >= groups_.shape(0)) continue;

                index_t group = groups_(row);
                if (group >= ngroups) continue;
                res_(group, c) += x_(j); // Different threads write to different columns, safe
            }
        }
    }
    return res;
}

// Sum groups for csr matrix (compressed sparse row)
py::array_t<double> cpp_sumGroups_csr(
    py::array_t<double> x,        // non-zero element values (by row)
    py::array_t<index_t> p,       // The start and end indices of each row's non-zero elements
    py::array_t<index_t> i,       // The column index corresponding to each non-zero element
    index_t ncol,                 // column number of the matrix
    index_t nrow,                 // row number of the matrix
    py::array_t<index_t> groups,  // The group index for each raw
    index_t ngroups,              // Total number of groups
    int nthreads                  // number of threads to use
) {
    // get read-only access to input data
    auto x_ = x.unchecked<1>();
    auto p_ = p.unchecked<1>();
    auto i_ = i.unchecked<1>();
    auto groups_ = groups.unchecked<1>();

    // creat result matrix
    py::array_t<double> res({ngroups, ncol});
    auto buf_res = res.request();
    double* ptr_res = (double*)buf_res.ptr;

    // initialize ptr_res to 0
    index_t total_size = ngroups * ncol;
    std::memset(ptr_res, 0, sizeof(double) * total_size);

    // Set number of threads
    const int max_threads = omp_get_max_threads();
    const int threads_to_use = std::clamp(
        nthreads > 0 ? nthreads : max_threads,
        1,
        max_threads
    );

    {
        py::gil_scoped_release release;

        std::vector<double> local_sums(static_cast<size_t>(threads_to_use) * total_size, 0.0);

        #pragma omp parallel num_threads(threads_to_use)
        {
            const int thread_id = omp_get_thread_num();
            double* local_buffer = local_sums.data() + thread_id * total_size;

            #pragma omp for schedule(dynamic)
            for (index_t r = 0; r < nrow; ++r) {
                const index_t group = groups_(r);
                if (group >= ngroups) continue; // skip invalid group

                for (index_t j = p_(r); j < p_(r + 1); ++j) {
                    const index_t col = i_(j);
                    if (col >= ncol) continue;
                    local_buffer[group * ncol + col] += x_(j);
                }
            }
        }

        #pragma omp parallel for num_threads(threads_to_use) schedule(static)
        for (index_t idx = 0; idx < total_size; ++idx) {
            double acc = 0.0;

            for (int t = 0; t < threads_to_use; ++t) {
                acc += local_sums[t * total_size + idx];
            }
            ptr_res[idx] = acc;
        }
    }
    return res;
}

py::array_t<double> cpp_sumGroups_csc_T(
    py::array_t<double> x,         // non-zero element values (by column)
    py::array_t<index_t> p,        // The start and end indices of each column's non-zero
    py::array_t<index_t> i,        // The raw index corresponding to each non-zero element
    index_t ncol,                  // column number of the matrix
    index_t nrow,                  // row number of the matrix
    py::array_t<index_t> groups,   // the group index for each column
    index_t ngroups,               // total number of groups
    int nthreads                   // number of threads to use
) {
    // Get read-only access to input data
    auto x_ = x.unchecked<1>();
    auto p_ = p.unchecked<1>();
    auto i_ = i.unchecked<1>();
    auto groups_ = groups.unchecked<1>();

    // Create result matrix
    py::array_t<double> res({ngroups, nrow});
    auto buf_res = res.request();
    double* ptr_res = (double*)buf_res.ptr;

    // Initialize res to 0
    index_t total_size = ngroups * nrow;
    std::memset(ptr_res, 0.0, sizeof(double) * total_size);

    // Set number of threads
    const int max_threads = omp_get_max_threads();
    const int threads_to_use = std::clamp(
        nthreads > 0 ? nthreads : max_threads,
        1,
        max_threads
    );

    {
        py::gil_scoped_release release;
        std::vector<double> local_sums(static_cast<size_t>(threads_to_use) * total_size, 0.0);

        #pragma omp parallel num_threads(threads_to_use)
        {
            const int thread_id = omp_get_thread_num();
            double* local_buffer = local_sums.data() + thread_id * total_size;

            #pragma omp for schedule(dynamic)
            for (index_t c = 0; c < ncol; ++c) {
                const index_t group = groups_(c);
                if (group >= ngroups) continue; // skip invalid group

                for (index_t j = p_(c); j < p_(c+1); ++j) {
                    const index_t row = i_(j);
                    if (row >= nrow) continue;
                    local_buffer[group * nrow + row] += x_(j);
                }
            }
        }

        #pragma omp parallel for num_threads(threads_to_use) schedule(static)
        for (index_t idx = 0; idx < total_size; ++idx) {
            double acc = 0.0;

            for (int t = 0; t < threads_to_use; ++t) {
                acc += local_sums[t * total_size + idx];
            }
            ptr_res[idx] = acc;
        }
    }
    return res;
}

py::array_t<double> cpp_sumGroups_csr_T(
    py::array_t<double> x,        // non-zero element values (by row)
    py::array_t<index_t> p,       // The start and end indices of each row's non-zero elements
    py::array_t<index_t> i,       // The column index corresponding to each non-zero element
    index_t nrow,                 // row number of the matrix
    py::array_t<index_t> groups,  // The group index for each column
    index_t ngroups,              // Total number of groups
    int nthreads                  // number of threads to use
) {
    // get read-only access to input data
    auto x_ = x.unchecked<1>();
    auto p_ = p.unchecked<1>();
    auto i_ = i.unchecked<1>();
    auto groups_ = groups.unchecked<1>();

    // create result matrix
    py::array_t<double, py::array::f_style> res({ngroups, nrow});
    auto res_ = res.mutable_unchecked<2>();

    // initialize res_ to 0
    std::memset(res_.mutable_data(0, 0), 0, sizeof(double) * ngroups * nrow);

    // Set number of threads
    const int max_threads = omp_get_max_threads();
    const int threads_to_use = std::clamp(
        nthreads > 0 ? nthreads : max_threads,
        1,
        max_threads
    );
    {
        py::gil_scoped_release release;

        #pragma omp parallel for num_threads(threads_to_use) schedule(dynamic)
        for (index_t r = 0; r < nrow; ++r) {
            for (index_t j = p_(r); j < p_(r + 1); ++j) {
                const index_t col = i_(j);
                if (col >= groups_.shape(0)) continue;
                const index_t group = groups_(col);
                if (group >= ngroups) continue;
                res_(group, r) += x_(j); // Different threads write to different columns, safe
            }
        }
    }
    return res;
}

py::array_t<double> cpp_sumGroups_dense(
    py::array_t<double> x,          // input dense matrix
    py::array_t<index_t> groups,    // group index for each row
    size_t ngroups                  // total number of groups
) {
    // get the read-only access to input data
    auto x_ = x.unchecked<2>();
    auto groups_ = groups.unchecked<1>();

    // get shape of the matrix
    size_t nrows = x_.shape(0);
    size_t ncols = x_.shape(1);

    // create res matrix
    std::vector<ssize_t> shape = {static_cast<ssize_t>(ngroups), static_cast<ssize_t>(ncols)};
    py::array_t<double> res(shape);
    auto res_ = res.mutable_unchecked<2>();

    std::memset(res_.mutable_data(0, 0), 0, sizeof(double) * ngroups * ncols);

    const py::buffer_info x_info = x.request();
    const double* x_ptr = static_cast<const double*>(x_info.ptr);

    // NumPy strides use bytes; CBLAS increments use elements.
    const Py_ssize_t row_stride = x_info.strides[0] / static_cast<Py_ssize_t>(sizeof(double));
    const Py_ssize_t col_stride = x_info.strides[1] / static_cast<Py_ssize_t>(sizeof(double));

    // traverse each row
    for (size_t r = 0; r < nrows; ++r) {
        size_t group = groups_(r);
        const double* x_row_ptr = x_ptr + static_cast<Py_ssize_t>(r) * row_stride;
        double* res_group_ptr = &res_(group, 0);
        cblas_daxpy(
            static_cast<int>(ncols),      // length of vector
            1.0,                          // alpha
            x_row_ptr,                    // input vector x
            static_cast<int>(col_stride), // incX
            res_group_ptr,                // output vector y
            1                             // incY
        );
    }

    return res;
}

py::array_t<double> cpp_sumGroups_dense_T(
    py::array_t<double> x,         // input dense matrix
    py::array_t<index_t> groups,   // group index for each column
    size_t ngroups                 // total number of groups
) {
    // get the read-only access to input data
    auto x_ = x.unchecked<2>();    // get the 2D array read-only accessor
    auto groups_ = groups.unchecked<1>(); // get the 1D array read-only accessor

    // get shape of the matrix
    size_t nrows = x_.shape(0);
    size_t ncols = x_.shape(1);

    // create res matrix
    std::vector<ssize_t> shape = {static_cast<ssize_t>(ngroups), static_cast<ssize_t>(nrows)};
    py::array_t<double> res(shape);
    auto res_ = res.mutable_unchecked<2>(); // Get a writable accessor for a two-dimensional array

    // initialize the result matrix to 0
    std::memset(res_.mutable_data(0, 0), 0, sizeof(double) * ngroups * nrows);

    const py::buffer_info x_info = x.request();
    const double* x_ptr = static_cast<const double*>(x_info.ptr);

    // NumPy strides use bytes; CBLAS increments use elements.
    const Py_ssize_t row_stride = x_info.strides[0] / static_cast<Py_ssize_t>(sizeof(double));
    const Py_ssize_t col_stride = x_info.strides[1] / static_cast<Py_ssize_t>(sizeof(double));

    // traverse each column
    for (size_t c = 0; c < ncols; ++c) {
        size_t group = groups_(c); // get the current column's group
        const double* x_col_ptr = x_ptr + static_cast<Py_ssize_t>(c) * col_stride;
        double* res_group_ptr = &res_(group, 0);
        cblas_daxpy(
            static_cast<int>(nrows),         // length of vector
            1.0,                             // alpha
            x_col_ptr,                       // input vector x
            static_cast<int>(row_stride),    // incX
            res_group_ptr,                   // output vector y
            1                                // incY
        );
    }
    return res;
}


py::array_t<count_t> cpp_nnzeroGroups_dense(
    py::array_t<double> x,         // input dense matrix
    py::array_t<index_t> groups,   // group index for each row
    index_t ngroups,               // total number of groups
    int nthreads                   // number of threads to use
) {
    // get the read-only access to input data
    auto x_ = x.unchecked<2>();    // get the 2D array read-only accessor
    auto groups_ = groups.unchecked<1>(); // get the 1D array read-only accessor

    // get shape of the matrix
    Py_ssize_t nrows = x_.shape(0);
    Py_ssize_t ncols = x_.shape(1);

    // create the result matrix
    std::vector<ssize_t> shape = {static_cast<ssize_t>(ngroups), static_cast<ssize_t>(ncols)};
    py::array_t<count_t, py::array::f_style> res(shape);
    auto res_ = res.mutable_unchecked<2>(); // get a writable accessor for a two-dimensional array

    // initialize the result matrix to 0
    std::memset(res_.mutable_data(0, 0), 0, sizeof(count_t) * ngroups * ncols);

    // Set number of threads
    const int max_threads = omp_get_max_threads();
    const int threads_to_use = std::clamp(
        nthreads > 0 ? nthreads : max_threads,
        1,
        max_threads
    );

    {
        py::gil_scoped_release release;

        // traverse each column
        #pragma omp parallel for num_threads(threads_to_use) schedule(dynamic)
        for (Py_ssize_t c = 0; c < ncols; ++c) {
            for (Py_ssize_t r = 0; r < nrows; ++r) {
                if (x_(r, c) != 0) {
                    res_(groups_(r), c) ++;
                }
            }
        }
    }

    return res;
}

py::array_t<count_t> cpp_nnzeroGroups_dense_T(
    py::array_t<double> x,         // input dense matrix
    py::array_t<index_t> groups, // group index for each column
    index_t ngroups,            // total number of groups
    int nthreads                   // number of threads to use
) {
    // get the read-only access to input data
    auto x_ = x.unchecked<2>();    // get the read-only accessor for 2D array
    auto groups_ = groups.unchecked<1>(); // get the read-only accessor for 1D array

    // get shape of the matrix
    Py_ssize_t nrows = x_.shape(0);
    Py_ssize_t ncols = x_.shape(1);

    // create result matrix
    std::vector<ssize_t> shape = {static_cast<ssize_t>(ngroups), static_cast<ssize_t>(nrows)};
    py::array_t<count_t, py::array::f_style> res(shape);
    auto res_ = res.mutable_unchecked<2>(); // get a writable accessor for a two-dimensional array

    // initialize the result matrix to 0
    std::memset(res_.mutable_data(0, 0), 0, sizeof(count_t) * ngroups * nrows);

    // Set number of threads
    const int max_threads = omp_get_max_threads();
    const int threads_to_use = std::clamp(
        nthreads > 0 ? nthreads : max_threads,
        1,
        max_threads
    );

    {
        py::gil_scoped_release release;

        // traverse each row
        #pragma omp parallel for num_threads(threads_to_use) schedule(dynamic)
        for (Py_ssize_t r = 0; r < nrows; ++r) {
            for (Py_ssize_t c = 0; c < ncols; ++c) {
                if (x_(r, c) != 0) {
                    res_(groups_(c), r)++;
                }
            }
        }
    }
    return res;
}

py::array_t<count_t> cpp_nnzeroGroups_csc(
    py::array_t<index_t> p,
    py::array_t<index_t> i,
    index_t ncol,
    py::array_t<index_t> groups,
    index_t ngroups,
    int nthreads
) {
    // read only access
    auto p_ = p.unchecked<1>();
    auto i_ = i.unchecked<1>();
    auto groups_ = groups.unchecked<1>();

    // create result matrix
    py::array_t<count_t, py::array::f_style> res({ngroups, ncol});
    auto res_ = res.mutable_unchecked<2>();

    // initialize to 0
    std::memset(res_.mutable_data(0, 0), 0, sizeof(count_t) * ngroups * ncol);

    // Set number of threads
    const int max_threads = omp_get_max_threads();
    const int threads_to_use = std::clamp(
        nthreads > 0 ? nthreads : max_threads,
        1,
        max_threads
    );

    {
        py::gil_scoped_release release;

        #pragma omp parallel for num_threads(threads_to_use) schedule(dynamic)
        for (index_t c = 0; c < ncol; ++c) {
            for (index_t j = p_(c); j < p_(c + 1); ++j) {
                index_t row = i_(j);
                if (row >= groups_.shape(0)) continue;
                index_t group = groups_(row);
                if (group >= ngroups) continue;
                res_(group, c)++;
            }
        }
    }
    return res;
}

py::array_t<count_t> cpp_nnzeroGroups_csr(
    py::array_t<index_t> p,
    py::array_t<index_t> i,
    index_t ncol,
    index_t nrow,
    py::array_t<index_t> groups,
    index_t ngroups,
    int nthreads
) {
    // get read-only access to input data
    auto p_ = p.unchecked<1>();
    auto i_ = i.unchecked<1>();
    auto groups_ = groups.unchecked<1>();

    // creat result matrix
    py::array_t<count_t> res({ngroups, ncol});
    auto buf_res = res.request();
    count_t* ptr_res = (count_t*)buf_res.ptr;

    // initialize res_ to 0
    index_t total_size = ngroups * ncol;
    std::memset(ptr_res, 0, sizeof(count_t) * total_size);

    // set number of threads
    const int max_threads = omp_get_max_threads();
    const int threads_to_use = std::clamp(
        nthreads > 0 ? nthreads : max_threads,
        1,
        max_threads
    );

    {
        py::gil_scoped_release release;
        std::vector<count_t> local_sums(static_cast<size_t>(threads_to_use) * total_size, 0);

        #pragma omp parallel num_threads(threads_to_use)
        {
            const int thread_id = omp_get_thread_num();
            count_t* local_buffer = local_sums.data() + thread_id * total_size;

            #pragma omp for schedule(dynamic)
            for (index_t r = 0; r < nrow; ++r) {
                const index_t group = groups_(r);
                if (group >= ngroups) continue; // skip invalid group

                for (index_t j = p_(r); j < p_(r + 1); ++j) {
                    const index_t col = i_(j);
                    if (col >= ncol) continue;
                    local_buffer[group * ncol + col] ++;
                }
            }
        }

        #pragma omp parallel for num_threads(threads_to_use) schedule(static)
        for (index_t idx = 0; idx < total_size; ++idx) {
            count_t acc = 0;

            for (int t = 0; t < threads_to_use; ++t) {
                acc += local_sums[t * total_size + idx];
            }
            ptr_res[idx] = acc;
        }
    }
    return res;
}

py::array_t<count_t> cpp_nnzeroGroups_csc_T(
    py::array_t<index_t> p,
    py::array_t<index_t> i,
    index_t ncol,
    index_t nrow,
    py::array_t<index_t> groups,
    index_t ngroups,
    int nthreads
) {
    // read only access
    auto p_ = p.unchecked<1>();
    auto i_ = i.unchecked<1>();
    auto groups_ = groups.unchecked<1>();

    // create result matrix
    py::array_t<count_t> res({ngroups, nrow});
    auto buf_res = res.request();
    count_t* ptr_res = (count_t*)buf_res.ptr;

    // initialize res to 0
    index_t total_size = ngroups * nrow;
    std::memset(ptr_res, 0, sizeof(count_t) * total_size);

    // Set number of threads
    const int max_threads = omp_get_max_threads();
    const int threads_to_use = std::clamp(
        nthreads > 0 ? nthreads : max_threads,
        1,
        max_threads
    );

    {
        py::gil_scoped_release release;
        std::vector<count_t> local_sums(static_cast<size_t>(threads_to_use) * total_size, 0);

        #pragma omp parallel num_threads(threads_to_use)
        {
            const int thread_id = omp_get_thread_num();
            count_t* local_buffer = local_sums.data() + thread_id * total_size;

            #pragma omp for schedule(dynamic)
            for (index_t c = 0; c < ncol; ++c) {
                const index_t group = groups_(c);
                if (group >= ngroups) continue; // skip invalid group

                for (index_t j = p_(c); j < p_(c+1); ++j) {
                    const index_t row = i_(j);
                    if (row >= nrow) continue;
                    local_buffer[group * nrow + row] ++;
                }
            }
        }
        #pragma omp parallel for schedule(static)
        for (index_t idx = 0; idx < total_size; ++idx) {
            count_t acc = 0;

            for (int t = 0; t < threads_to_use; ++t) {
                acc += local_sums[t * total_size + idx];
            }
            ptr_res[idx] = acc;
        }
    }
    return res;
}

py::array_t<count_t> cpp_nnzeroGroups_csr_T(
    py::array_t<index_t> p,
    py::array_t<index_t> i,
    index_t nrow,
    py::array_t<index_t> groups,
    index_t ngroups,
    int nthreads
) {
    auto p_ = p.unchecked<1>();
    auto i_ = i.unchecked<1>();
    auto groups_ = groups.unchecked<1>();

    // create result matrix
    py::array_t<count_t, py::array::f_style> res({ngroups, nrow});
    auto res_ = res.mutable_unchecked<2>();

    std::memset(res_.mutable_data(0, 0), 0, sizeof(count_t) * ngroups * nrow);

    // Set number of threads
    const int max_threads = omp_get_max_threads();
    const int threads_to_use = std::clamp(
        nthreads > 0 ? nthreads : max_threads,
        1,
        max_threads
    );
    {
        py::gil_scoped_release release;

        #pragma omp parallel for num_threads(threads_to_use) schedule(dynamic)
        for (index_t r = 0; r < nrow; ++r) {
            for (index_t j = p_(r); j < p_(r + 1); ++j) {
                const index_t col = i_(j);
                if (col >= groups_.shape(0)) continue;
                const index_t group = groups_(col);
                if (group >= ngroups) continue;
                res_(group, r) ++; // Different threads write to different columns, safe
            }
        }
    }

    return res;
}

std::vector<std::vector<float>> cpp_rank_matrix_csc(
    py::array_t<double> x_in,         // CSC format: non-zero values
    py::array_t<index_t> p_in,        // CSC format: column pointers
    py::array_t<double> rank_data_out,// Output: array to store ranks
    index_t nrow,
    index_t ncol,
    int nthreads
) {
	// get ptr for input data
	auto p_ = p_in.unchecked<1>();

	// access raw data ptrs
	py::buffer_info buf_x = x_in.request();
    double* ptr_x = static_cast<double*>(buf_x.ptr);

    // prepare output buffer
    py::buffer_info buf_out = rank_data_out.request();
    double* ptr_out = static_cast<double*>(buf_out.ptr);

    if (buf_out.size != buf_x.size) {
        throw std::runtime_error("rank_data_out size must match x_in size");
    }
    std::vector<std::vector<float>> ties(ncol);

    // set number of threads
    const int max_threads = omp_get_max_threads();
    const int threads_to_use = std::clamp(
        nthreads > 0 ? nthreads : max_threads,
        1,
        max_threads
    );

    {
    	py::gil_scoped_release release;

    	#pragma omp parallel num_threads(threads_to_use)
    	{
    		std::vector<std::pair<double, index_t>> workspace;
    		#pragma omp for schedule(dynamic)
    		for (index_t col = 0; col < ncol; ++col) {
    			index_t start_idx = p_(col);
    			index_t end_idx = p_(col + 1);

    			// all-zero column
    			if (start_idx == end_idx) {
    				ties[col].push_back(static_cast<float>(nrow));
            		continue;
    			}
    			// number of zeros in the column
    			index_t n_zero = nrow - (end_idx - start_idx);
    			// allocate for workspace
    			workspace.clear();
    			workspace.reserve(end_idx - start_idx);
    			for (index_t i = start_idx; i < end_idx; ++i) {
            		workspace.emplace_back(ptr_x[i], i);
        		}
    			// sort by value to find ties
        		std::sort(workspace.begin(), workspace.end());
        		double rank_sum = 0;
        		int n = 1;
        		size_t i;

        		for (i = 1; i < workspace.size(); ++i) {
        			if (!is_tied(workspace[i].first, workspace[i - 1].first)) {
        				double avg_rank = (rank_sum / n) + 1 + n_zero;
        				for (index_t j = 0; j < n; ++j) {
        					ptr_out[workspace[i - 1 - j].second] = avg_rank;
        				}
        				if (n > 1) ties[col].push_back(static_cast<float>(n));
        				rank_sum = i;
        				n = 1;
        			} else {
        				rank_sum += i;
        				n ++;
        			}
        		}
        		// the last group
        		double avg_rank = (rank_sum / n) + 1 + n_zero;
        		for (index_t j = 0; j < n; ++j) {
        			ptr_out[workspace[i - 1 - j].second] = avg_rank;
        		}
        		if (n > 1) ties[col].push_back(static_cast<float>(n));
        		// Record zero count as the last element in ties
                ties[col].push_back(static_cast<float>(n_zero));
    		}
    	}
    }
    return ties;
}

py::dict cpp_rank_matrix_dense(
    py::array_t<double> input,
    int nthreads
) {
    // Get buffer info
    auto buf = input.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Input must be a 2D array");

    // Get dimensions
    Py_ssize_t n_rows = buf.shape[0];
    Py_ssize_t n_cols = buf.shape[1];

    // Create a copy of the input array to work with
    auto result = py::array_t<double>(buf.shape);
    auto result_buf = result.mutable_unchecked<2>();
    auto input_buf = input.unchecked<2>();

    // Prepare to store tie information
    std::vector<std::list<float>> ties(n_cols);

    // Set number of threads
    const int max_threads = omp_get_max_threads();
    const int threads_to_use = std::clamp(
        nthreads > 0 ? nthreads : max_threads,
        1,
        max_threads
    );

    {
        py::gil_scoped_release release;

        // Process each column
        #pragma omp parallel for num_threads(threads_to_use) schedule(dynamic)
        for (Py_ssize_t c = 0; c < n_cols; c++) {
            // Create a vector to sort values (value, original index)
            std::vector<std::pair<double, Py_ssize_t>> v_sort;
            v_sort.reserve(n_rows);

            for (Py_ssize_t i = 0; i < n_rows; i++) {
                v_sort.emplace_back(input_buf(i, c), i);
            }

            // Sort by value (ascending)
            std::sort(v_sort.begin(), v_sort.end());

            // Calculate ranks
            double rank_sum = 0;
            int n = 1;
            size_t i;

            for (i = 1; i < v_sort.size(); i++) {
                // Check for equality with tolerance for floating point
                if (!is_tied(v_sort[i].first, v_sort[i-1].first)) {
                    // Assign rank to previous tied values
                    double rank = (rank_sum / n) + 1;
                    for (int j = 0; j < n; j++) {
                        result_buf(v_sort[i-1-j].second, c) = rank;
                    }

                    // Reset for next group
                    rank_sum = i;
                    if (n > 1) ties[c].push_back(static_cast<float>(n));
                    n = 1;
                } else {
                    // Continue counting ties
                    rank_sum += i;
                    n++;
                }
            }

            // Handle the last group of values
            double rank = (rank_sum / n) + 1;
            for (int j = 0; j < n; j++) {
                result_buf(v_sort[i-1-j].second, c) = rank;
            }
            if (n > 1) ties[c].push_back(static_cast<float>(n));
        }
    }
    // Convert ties to a format that can be returned to Python
    py::list py_ties(n_cols);
    for (size_t c = 0; c < n_cols; c++) {
        py::list col_ties;
        for (float tie_size : ties[c]) {
            col_ties.append(tie_size);
        }
        py_ties[c] = col_ties;
    }

    return py::dict(
        "X_ranked"_a = result,
        "ties"_a = py_ties
    );
}

void cpp_group_rank_csc(
    py::array_t<double> x_in,           // csc data
    py::array_t<double> rank_data_out,  // output rank data
    py::array_t<index_t> p_in,          // indptr for col
    py::array_t<index_t> i_in,          // indice for row
    py::array_t<index_t> groups,        // group id for each cell
    index_t n_cols,                     // number of genes
    int nthreads                       // OpenMP threads
) {

    auto p_ = p_in.unchecked<1>();
    auto i_ = i_in.unchecked<1>();
    auto groups_ = groups.unchecked<1>();

    py::buffer_info buf_x = x_in.request();
    double *ptr_x = static_cast<double*>(buf_x.ptr);

    py::buffer_info buf_out = rank_data_out.request();
    double* ptr_out = static_cast<double*>(buf_out.ptr);

    if (buf_out.size != buf_x.size) {
        throw std::runtime_error("rank_data_out size must match x_in size");
    }

    // set number of threads
    const int max_threads = omp_get_max_threads();
    const int threads_to_use = std::clamp(
        nthreads > 0 ? nthreads : max_threads,
        1,
        max_threads
    );

    {
        py::gil_scoped_release release;

        #pragma omp parallel num_threads(threads_to_use)
        {
            std::vector<Entry> col_entries;
            #pragma omp for schedule(dynamic)
            for (index_t col = 0; col < n_cols; ++col) {
                index_t start = p_(col);
                index_t end = p_(col + 1);

                if (start == end) continue;

                // collect all non-zero information
                col_entries.clear();
                col_entries.reserve(end - start);
                for (index_t idx = start; idx < end; ++idx) {
                    index_t row = i_(idx);
                    index_t group = groups_(row);
                    double value = ptr_x[idx];
                    col_entries.push_back({idx, group, value});
                }

                // sort by value
                std::sort(col_entries.begin(), col_entries.end());

                // calc rank and written in the results
                // col_entries: [group_0_low, group_0_medium, group_0_large,
                //               group_1_low, group_1_medium, group_1_large, ...]
                if (!col_entries.empty()) {
                    index_t current_group = col_entries[0].group_id;
                    double current_rank = 1.0;
                    for (const auto& entry : col_entries) {
                        if (current_group != entry.group_id) {
                            current_group = entry.group_id;
                            current_rank = 1.0;
                        }
                        ptr_out[entry.data_index] = current_rank;
                        current_rank ++;
                    }
                }
            }
        }
    }
}

void cpp_group_rank_dense(
    py::array_t<double> x_in,               // dense data
    py::array_t<double> rank_data_out,      // output_rank_data
    py::array_t<index_t> groups,            // group id for each row
    int nthreads                            // OpenMP threads
) {
    auto x_ = x_in.unchecked<2>();
    auto out_ = rank_data_out.mutable_unchecked<2>();
    auto groups_ = groups.unchecked<1>();

    index_t n_rows = static_cast<index_t>(x_.shape(0));
    index_t n_cols = static_cast<index_t>(x_.shape(1));

    // set number of threads
    const int max_threads = omp_get_max_threads();
    const int threads_to_use = std::clamp(
        nthreads > 0 ? nthreads : max_threads,
        1,
        max_threads
    );

    {
        py::gil_scoped_release release;

        #pragma omp parallel num_threads(threads_to_use)
        {
            std::vector<DenseEntry> col_entries;
            col_entries.reserve(n_rows);

            #pragma omp for schedule(dynamic)
            for (index_t col = 0; col < n_cols; ++col) {
                col_entries.clear();

                for (index_t row = 0; row < n_rows; ++row) {
                    double val = x_(row, col);

                    if (std::abs(val) > 1e-15) {
                        index_t grp = groups_(row);
                        col_entries.push_back({row, grp, val});
                    }
                }
                std::sort(col_entries.begin(), col_entries.end());
                if (!col_entries.empty()) {
                    index_t current_group = col_entries[0].group_id;
                    double current_rank = 1.0;
                    for (const auto& entry : col_entries) {
                        if (current_group != entry.group_id) {
                            current_group = entry.group_id;
                            current_rank = 1.0;
                        }
                        out_(entry.row_idx, col) = current_rank;
                        current_rank ++;
                    }
                }
            }
        }
    }
}

// export module
PYBIND11_MODULE(_fastwilcoxon, m) {

    m.doc() = "High-performance matrix operations for wilcoxon based statistics and ranking";

    m.def("cpp_sumGroups_csc", &cpp_sumGroups_csc,
          "Sum groups for CSC sparse matrix (groups by rows)",
          py::arg("x"),
          py::arg("p"),
          py::arg("i"),
          py::arg("ncol"),
          py::arg("groups"),
          py::arg("ngroups"),
          py::arg("nthreads"));

    m.def("cpp_sumGroups_csr", &cpp_sumGroups_csr,
          "Sum groups for CSR sparse matrix (groups by cols)",
          py::arg("x"),
          py::arg("p"),
          py::arg("i"),
          py::arg("ncol"),
          py::arg("nrow"),
          py::arg("groups"),
          py::arg("ngroups"),
          py::arg("nthreads"));

    m.def("cpp_sumGroups_csc_T", &cpp_sumGroups_csc_T,
          "Sum groups for transposed CSC sparse matrix (groups by columns)",
          py::arg("x"),
          py::arg("p"),
          py::arg("i"),
          py::arg("ncol"),
          py::arg("nrow"),
          py::arg("groups"),
          py::arg("ngroups"),
          py::arg("nthreads"));

    m.def("cpp_sumGroups_csr_T", &cpp_sumGroups_csr_T,
          "Sum groups for transposed CSR sparse matrix (groups by rows)",
          py::arg("x"),
          py::arg("p"),
          py::arg("i"),
          py::arg("nrow"),
          py::arg("groups"),
          py::arg("ngroups"),
          py::arg("nthreads"));

    m.def("cpp_sumGroups_dense", &cpp_sumGroups_dense,
          "Sum values by group for dense matrix (groups by rows)",
          py::arg("x"),
          py::arg("groups"),
          py::arg("ngroups"));

    m.def("cpp_sumGroups_dense_T", &cpp_sumGroups_dense_T,
          "Sum values by group for transposed dense matrix (groups by columns)",
          py::arg("x"),
          py::arg("groups"),
          py::arg("ngroups"));

    m.def("cpp_nnzeroGroups_dense", &cpp_nnzeroGroups_dense,
          "Count non-zero elements by group for dense matrix (groups by rows)",
          py::arg("x"),
          py::arg("groups"),
          py::arg("ngroups"),
          py::arg("nthreads"));

    m.def("cpp_nnzeroGroups_dense_T", &cpp_nnzeroGroups_dense_T,
          "Count non-zero elements by group for transposed dense matrix (groups by cols)",
          py::arg("x"),
          py::arg("groups"),
          py::arg("ngroups"),
          py::arg("nthreads"));

    m.def("cpp_nnzeroGroups_csc", &cpp_nnzeroGroups_csc,
          "Count non-zero elements by group for CSC sparse matrix (groups by rows)",
          py::arg("p"),
          py::arg("i"),
          py::arg("ncol"),
          py::arg("groups"),
          py::arg("ngroups"),
          py::arg("nthreads"));

    m.def("cpp_nnzeroGroups_csr", &cpp_nnzeroGroups_csr,
          "Count non-zero elements by group for CSR sparse matrix (groups by cols)",
          py::arg("p"),
          py::arg("i"),
          py::arg("ncol"),
          py::arg("nrow"),
          py::arg("groups"),
          py::arg("ngroups"),
          py::arg("nthreads"));

    m.def("cpp_nnzeroGroups_csc_T", &cpp_nnzeroGroups_csc_T,
          "Count non-zero elements by group for transposed CSC sparse matrix (groups by columns)",
          py::arg("p"),
          py::arg("i"),
          py::arg("ncol"),
          py::arg("nrow"),
          py::arg("groups"),
          py::arg("ngroups"),
          py::arg("nthreads"));

    m.def("cpp_nnzeroGroups_csr_T", &cpp_nnzeroGroups_csr_T,
      "Count non-zero elements by group for transposed CSR sparse matrix (groups by rows)",
      py::arg("p"),
      py::arg("i"),
      py::arg("nrow"),
      py::arg("groups"),
      py::arg("ngroups"),
      py::arg("nthreads"));

    m.def("cpp_rank_matrix_csc", &cpp_rank_matrix_csc,
          "Rank columns of a sparse matrix in CSC format",
          py::arg("x_in"),
          py::arg("p_in"),
          py::arg("rank_data_out"),
          py::arg("nrow"),
          py::arg("ncol"),
          py::arg("nthreads"));

    m.def("cpp_rank_matrix_dense", &cpp_rank_matrix_dense,
          "Compute column-wise ranks for dense matrix",
          py::arg("input"),
          py::arg("nthreads"));

    m.def("cpp_group_rank_csc", &cpp_group_rank_csc,
          "Compute group-wise ranks for CSC sparse matrix",
          py::arg("x_in"),
          py::arg("rank_data_out"),
          py::arg("p_in"),
          py::arg("i_in"),
          py::arg("groups"),
          py::arg("n_cols"),
          py::arg("nthreads"));

    m.def("cpp_group_rank_dense", &cpp_group_rank_dense,
          "Compute group-wise ranks for dense matrix",
          py::arg("x_in"),
          py::arg("rank_data_out"),
          py::arg("groups"),
          py::arg("nthreads"));
}
