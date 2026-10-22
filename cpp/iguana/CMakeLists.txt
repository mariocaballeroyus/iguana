add_library(iguana
    basis/bspline.cpp
    basis/tensor_bspline.cpp
    basis/tensor_nurbs.cpp
    grid/hierarchical_grid.cpp
    grid/hierarchical_grid_iterator.cpp
    grid/knot_vector.cpp
    grid/tensor_grid.cpp
    grid/tensor_grid_iterator.cpp
    embedding/cell_classification.cpp
    embedding/embedded_boundary.cpp
    embedding/face_selection.cpp
    embedding/nurbs/grid_crossings.cpp
    embedding/nurbs/volume_fractions.cpp
    embedding/surrogate_boundary.cpp
    fspace/dof_map.cpp
    fspace/function_space.cpp
    fspace/mixed_function_space.cpp
    geometry/boundary.cpp
    geometry/boundary_projection.cpp
    geometry/field_interpolation.cpp
    geometry/normal_coordinates.cpp
    geometry/patch.cpp
    quadrature/box_rule.cpp
    quadrature/gauss_legendre/gauss_legendre.cpp
    quadrature/gauss_legendre/gauss_legendre_tables.cpp
    quadrature/moment_fitting/moment_fitting.cpp
    quadrature/moment_fitting/moments.cpp
    quadrature/quadrature_boundary.cpp
    quadrature/quadrature_domain.cpp
    quadrature/quadrature_skeleton.cpp
    quadrature/xiao_gimbutas/xiao_gimbutas.cpp
    quadrature/xiao_gimbutas/xiao_gimbutas_tables.cpp
    assembly/element_values.cpp
    element/poisson/poisson_element.cpp
    element/poisson/poisson_normal_series.cpp
    element/poisson/poisson_shift.cpp
    element/stokes/stokes_element.cpp
    element/stokes/stokes_stabilized_element.cpp
    condition/penalty_condition.cpp
    condition/nitsche_condition.cpp
    condition/skeleton_nitsche_condition.cpp
    condition/neumann_condition.cpp
    condition/skeleton_ghost_penalty.cpp)

add_library(IGUANA::iguana ALIAS iguana)

target_compile_features(iguana PUBLIC cxx_std_20)

set_target_properties(iguana PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)

target_compile_options(iguana PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4;/MP>)

target_include_directories(iguana PUBLIC ${PROJECT_SOURCE_DIR}/cpp)

target_link_libraries(iguana PUBLIC Eigen3::Eigen)

# The assembler writes into PETSc matrices, and everything else stays free of
# PETSc, so that a build without it still evaluates and integrates
if(IGUANA_WITH_PETSC)
    target_sources(iguana PRIVATE assembly/assembler.cpp)

    target_compile_definitions(iguana PUBLIC IGUANA_WITH_PETSC)

    target_link_libraries(iguana PUBLIC PkgConfig::PETSc MPI::MPI_CXX)
endif()
