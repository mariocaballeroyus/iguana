# IGUANA: IsoGeometric Unfitted ANAlysis

[![CI](https://github.com/mariocaballeroyus/iguana/actions/workflows/ci.yml/badge.svg)](https://github.com/mariocaballeroyus/iguana/actions/workflows/ci.yml)

*IGUANA* is a C++ library for finite element analysis using isogeometric, unfitted (immersed) discretizations.

```
                                  __ \/_
                                 (' \`\
 _                           _\, \ \\/   
(_)                           /`\/\ \\                         
 _  __ _ _   _  __ _ _ __   __ _   \ \\
| |/ _` | | | |/ _` | '_ \ / _` |   \ \\/\/_
| | (_| | |_| | (_| | | | | (_| |   /\ \\'\
|_|\__, |\__,_|\__,_|_| |_|\__,_| __\ `\\\
    __/ |                          /|`  `\\
   |___/                                  \\    ,
                                           `---'
```

## Building

IGUANA depends on Eigen and Catch2, both included as submodules:

```bash
git clone --recurse-submodules https://github.com/mariocaballeroyus/iguana.git
cd iguana
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build
```

If the repository was cloned without them, the submodules are fetched with:

```bash
git submodule update --init --recursive
```

Requires a C++20 compiler and CMake 3.21 or newer.
