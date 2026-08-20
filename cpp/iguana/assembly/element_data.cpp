/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "element_data.hpp"

#include "iguana/geometry/differential.hpp"

namespace iguana
{

template<int d, std::floating_point T>
void ElementData<d, T>::reinit(const TensorDomainIterator<d, T>& element)
{
    const TensorProductBSpline<d, T>& basis = domain_->basis();

    // The points and the actives, needed by every consumer
    rule_->map_to(element.start(), element.end(), points, weights);
    basis.active_on_element(element.index(), actives);

    // The quantity ladder, each step from the ones before it; the
    // basis evaluation holds the values as its order 0
    if (order_ >= 0)
        basis.eval_ders_on_element(element.first_active(), points,
                                   order_, values);

    if (flags_ & update_positions)
        patch_->position(actives, values[0], positions);

    if (flags_ & update_tangents)
        patch_->tangents(actives, values[1], tangents);

    if (flags_ & update_metric)
        iguana::metric(tangents, metric);

    if (flags_ & update_metric_inverse)
        iguana::metric_inverse<d>(metric, metric_inverse);

    if (flags_ & update_measure)
        measure<d>(metric, measures);

    if (flags_ & update_weights)
        weights.array() *= measures.array();
}

template class ElementData<1, double>;
template class ElementData<2, double>;
template class ElementData<3, double>;

} // namespace iguana
