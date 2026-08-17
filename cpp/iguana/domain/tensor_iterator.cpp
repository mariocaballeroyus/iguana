/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "tensor_iterator.hpp"

namespace iguana
{

template class TensorDomainIterator<1, double>;
template class TensorDomainIterator<2, double>;
template class TensorDomainIterator<3, double>;

} // namespace iguana
