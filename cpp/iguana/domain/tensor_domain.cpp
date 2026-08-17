/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "tensor_domain.hpp"

namespace iguana
{

template class TensorDomain<1, double>;
template class TensorDomain<2, double>;
template class TensorDomain<3, double>;

} // namespace iguana
