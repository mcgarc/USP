# USP

A simple Python 3 simulation of particles trapped by current-carrying wires.

Released under GNU 3 License

Written by Cameron McGarry

## About

USP is software for simulating particles trapped in a magnetic potential. It
has been designed for simulating the motion of molecules in and around chip
traps. As such it includes tools that can be used to simulate common traps,
including H-, U- and Z-traps. Arbitrary layouts of straight wires can also be
simulated, as can quadrupoles, and indeed any function specifying a trapping
potential can be used.  There are also tools for investigating the changing of
potentials by ramping of currents, bias fields and so forth.


## Examples

Some example scripts are included to demonstrate intended use cases. It is
recommended that simulation output is saved to a file and analysed separately.

## Potential improvements

The project has been designed for the work I undertook at Imperial College
London, which focused on a microfabricated chip trap for CaF molecules. Minor
changes will need to be made to the source code to simulate atoms or other
species. Implementing seamless integration will be harder.

The simulation makes an assumption that the motion is adiabatic, which should
be taken into account for simulations, and could be removed in future
iterations.

The integration method used is not necessarily the fastest. It may be possible
to improve runtime by using a different method or using custom code.

If you intend to use this codebase, or would like to make improvements then
feel free to get in touch.

## Install

1. Pull files to local repository
2. (Recommended) Create and activate virtual environment
3. `pip install -r requirements`
4. `cc -fPIC -shared -o USP/wire_segment_fn.so USP/wire_segment_fn.c`
5. Enjoy!

## Test

Run `test.py`, or for coverage use `sh run_coverage.sh`

## Name

Originally titled _Untitled Simulation Project_, and briefly referred to as
_USP Simulates Particles_, the name of this software is now simply _USP_ . I am
sorry for not choosing a more descriptive name.

