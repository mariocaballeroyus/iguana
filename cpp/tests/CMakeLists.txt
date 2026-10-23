add_executable(iguana_tests
    basis/test_bspline.cpp
    basis/test_tensor_bspline.cpp
    basis/test_tensor_nurbs.cpp
    grid/test_hierarchical_grid.cpp
    grid/test_knot_vector.cpp
    grid/test_tensor_grid_iterator.cpp
    embedding/facets/test_clipper.cpp
    embedding/facets/test_inside.cpp
    embedding/facets/test_slicer.cpp
    embedding/nurbs/test_axis_crossings.cpp
    embedding/nurbs/test_grid_crossings.cpp
    embedding/nurbs/test_volume_fractions.cpp
    embedding/test_embedded_boundary.cpp
    embedding/test_face_selection.cpp
    embedding/test_surrogate_boundary.cpp
    fspace/test_dof_map.cpp
    fspace/test_function_space.cpp
    geometry/test_boundary.cpp
    geometry/test_field_interpolation.cpp
    geometry/test_patch.cpp
    quadrature/gauss_legendre/test_gauss_legendre.cpp
    quadrature/moment_fitting/test_moment_fitting.cpp
    quadrature/moment_fitting/test_moments.cpp
    quadrature/test_quadrature_boundary.cpp
    quadrature/test_quadrature_domain.cpp
    quadrature/test_quadrature_skeleton.cpp
    quadrature/xiao_gimbutas/test_xiao_gimbutas.cpp
    assembly/test_element_values.cpp
    element/poisson/test_poisson_element.cpp
    element/poisson/test_poisson_shift.cpp
    element/poisson/test_poisson_traces.cpp
    element/stokes/test_stokes_element.cpp
    utils/test_bernstein.cpp
    utils/test_legendre.cpp
    utils/test_taylor_series.cpp)

target_compile_features(iguana_tests PRIVATE cxx_std_20)

target_link_libraries(iguana_tests PRIVATE
    IGUANA::iguana
    Catch2::Catch2WithMain)

if(IGUANA_WITH_PETSC)
    target_sources(iguana_tests PRIVATE
        petsc_listener.cpp
        assembly/test_assembler.cpp)
endif()

add_test(NAME iguana_tests COMMAND iguana_tests)
