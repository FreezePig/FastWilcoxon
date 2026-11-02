cmake_minimum_required(VERSION 3.15)
project(FastWilcoxon LANGUAGES CXX)

# --- 1. Basic Set ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- 2. Find Python Environment ---
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# oneMKL static CPU archives are build-only Python dependencies on Linux x86_64.
# Keep one BLAS provider per extension: MKL builds do not link OpenBLAS as well.
set(FASTWILCOXON_MKL_DEFAULT OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(FASTWILCOXON_MKL_DEFAULT ON)
endif()
option(FASTWILCOXON_ENABLE_MKL "Build the oneMKL CSR-to-CSC backend" ${FASTWILCOXON_MKL_DEFAULT})
if(FASTWILCOXON_ENABLE_MKL AND NOT FASTWILCOXON_MKL_DEFAULT)
    message(FATAL_ERROR "The bundled static oneMKL build supports Linux x86_64 only")
endif()

# --- 3. Find Openmp ---
if(APPLE)
    # 1. Find brew installed libomp
    find_program(BREW_CMD brew)
    if(BREW_CMD)
        execute_process(COMMAND ${BREW_CMD} --prefix libomp
                OUTPUT_VARIABLE BREW_OPENMP_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()

    # 2. Path fallback mechanism
    if(NOT EXISTS "${BREW_OPENMP_PREFIX}")
        if(EXISTS "/opt/homebrew/opt/libomp")
            set(BREW_OPENMP_PREFIX "/opt/homebrew/opt/libomp")
        elseif(EXISTS "/usr/local/opt/libomp")
            set(BREW_OPENMP_PREFIX "/usr/local/opt/libomp")
        endif()
    endif()

    if(BREW_OPENMP_PREFIX)
        message(STATUS "macOS: Found OpenMP prefix at ${BREW_OPENMP_PREFIX}")

        # Force setting OpenMP_ROOT helps FindOpenMP locate the header file.
        set(OpenMP_ROOT "${BREW_OPENMP_PREFIX}")

        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${BREW_OPENMP_PREFIX}/include")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${BREW_OPENMP_PREFIX}/include")

        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${BREW_OPENMP_PREFIX}/lib/libomp.dylib")

        include_directories("${BREW_OPENMP_PREFIX}/include")
        link_directories("${BREW_OPENMP_PREFIX}/lib")
    else()
        message(WARNING "macOS: libomp not found. Please run 'brew install libomp'")
    endif()
endif()

find_package(OpenMP REQUIRED)

if(OpenMP_FOUND AND APPLE)
    if(NOT OpenMP_CXX_FLAGS MATCHES "-Xpreprocessor")
        message(STATUS "macOS: Patching OpenMP flags for AppleClang")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp ${OpenMP_CXX_FLAGS}")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp ${OpenMP_C_FLAGS}")
    endif()
endif()

if(OpenMP_FOUND)
    message(STATUS "OpenMP Found. Version: ${OpenMP_CXX_VERSION}")
else()
    message(FATAL_ERROR "Could NOT find OpenMP. Please ensure libomp is installed.")
endif()

# ---4. OpenBLAS setup ---
if(NOT FASTWILCOXON_ENABLE_MKL)
if(WIN32)
    # detect x64
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(OPENBLAS_ARCH_DIR "x64")
        message(STATUS "Building for Windows 64-bit")
    else()
        message(FATAL_ERROR "Not supported currently: 32-bit Windows build.")
    endif()

    set(OPENBLAS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/OpenBLAS/${OPENBLAS_ARCH_DIR}")
    
    if(NOT EXISTS "${OPENBLAS_ROOT}")
        message(FATAL_ERROR "OpenBLAS not found at: ${OPENBLAS_ROOT}. Please ensure external/OpenBLAS/x64 and x86 exist.")
    endif()

    include_directories("${OPENBLAS_ROOT}/include")
    link_directories("${OPENBLAS_ROOT}/lib")
    
    # Try linking the static library libopenblas.a (if it exists) to avoid DLL dependencies.
    # If only .lib or .dll.a files are available, you will need to use delvewheel to fix them after packaging.
    find_library(BLAS_LIB
            NAMES libopenblas openblas
            PATHS "${OPENBLAS_ROOT}/lib"
            NO_DEFAULT_PATH
    )
    if(NOT BLAS_LIB)
        message(FATAL_ERROR "Could not find OpenBLAS library in ${OPENBLAS_ROOT}/lib")
    endif()
    set(BLAS_LIBRARIES ${BLAS_LIB})
else()
    # 1. Find any BLAS library (OpenBLAS/MKL/Atlas etc.)
    # Using CMake FindBLAS.cmake
    find_package(BLAS REQUIRED)

    # 2. Find header cblas.h
    if(NOT BLAS_INCLUDE_DIRS)
        find_path(BLAS_INCLUDE_DIRS cblas.h
                PATHS
                /usr/include/openblas
                /usr/local/include/openblas
                /opt/homebrew/include
                /opt/homebrew/opt/openblas/include
                /usr/include
                /usr/local/include
        )
    endif()

    if(BLAS_INCLUDE_DIRS)
        include_directories(${BLAS_INCLUDE_DIRS})
        message(STATUS "Found BLAS headers in: ${BLAS_INCLUDE_DIRS}")
    else()
        message(WARNING "Could NOT find cblas.h. Compilation might fail if headers are missing.")
    endif()
    message(STATUS "Found BLAS libraries: ${BLAS_LIBRARIES}")
endif()
endif()

# --- 5. define target module ---
pybind11_add_module(_fastwilcoxon main.cpp)

# NumPy headers (only if needed)
target_include_directories(_fastwilcoxon PRIVATE ${NUMPY_INCLUDE_DIR})

# BLAS: the three static archives contain MKL's CPU dispatch kernels as well.
# Unlike dynamic MKL, no additional ISA libraries need to be found at runtime.
if(FASTWILCOXON_ENABLE_MKL)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_mkl_static.py"
        OUTPUT_VARIABLE FASTWILCOXON_MKL_FILES
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE FASTWILCOXON_MKL_RESULT
    )
    if(NOT FASTWILCOXON_MKL_RESULT EQUAL 0)
        message(FATAL_ERROR "oneMKL static build dependencies were not found in the build environment")
    endif()
    list(GET FASTWILCOXON_MKL_FILES 0 MKL_INCLUDE_DIR)
    list(GET FASTWILCOXON_MKL_FILES 1 MKL_ILP64_ARCHIVE)
    list(GET FASTWILCOXON_MKL_FILES 2 MKL_GNU_THREAD_ARCHIVE)
    list(GET FASTWILCOXON_MKL_FILES 3 MKL_CORE_ARCHIVE)
    list(GET FASTWILCOXON_MKL_FILES 4 MKL_LICENSE_FILE)
    find_package(Threads REQUIRED)
    target_compile_definitions(_fastwilcoxon PRIVATE FASTWILCOXON_WITH_MKL MKL_ILP64)
    target_include_directories(_fastwilcoxon PRIVATE "${MKL_INCLUDE_DIR}")
    target_link_libraries(_fastwilcoxon PRIVATE
        "-Wl,--start-group"
        "${MKL_ILP64_ARCHIVE}"
        "${MKL_GNU_THREAD_ARCHIVE}"
        "${MKL_CORE_ARCHIVE}"
        "-Wl,--end-group"
        Threads::Threads m "${CMAKE_DL_LIBS}"
    )
    # Keep bundled static BLAS symbols private to this extension.
    target_link_options(_fastwilcoxon PRIVATE "-Wl,--exclude-libs,ALL")
    install(FILES "${MKL_LICENSE_FILE}" DESTINATION licenses RENAME oneMKL-LICENSE.txt)
elseif(WIN32)
    target_include_directories(_fastwilcoxon PRIVATE "${OPENBLAS_ROOT}/include")
    target_link_libraries(_fastwilcoxon PRIVATE "${OPENBLAS_ROOT}/lib/libopenblas.dll.a")
else()
    target_include_directories(_fastwilcoxon PRIVATE ${BLAS_INCLUDE_DIRS})
    target_link_libraries(_fastwilcoxon PRIVATE ${BLAS_LIBRARIES})
endif()

# OpenMP
target_link_libraries(_fastwilcoxon PRIVATE OpenMP::OpenMP_CXX)

# compilation options
if(MSVC)
    target_compile_options(_fastwilcoxon PRIVATE /W3 /O2)
else()
    target_compile_options(_fastwilcoxon PRIVATE -O3 -Wall -fvisibility=hidden)
endif()

# macOS RPATH
if(APPLE)
    # Leave space for the shared OpenMP install name written during wheel repair.
    target_link_options(_fastwilcoxon PRIVATE "-Wl,-headerpad_max_install_names")
    set_target_properties(_fastwilcoxon PROPERTIES
            INSTALL_RPATH "@loader_path"
            BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

install(TARGETS _fastwilcoxon DESTINATION .)
