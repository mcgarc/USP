# Untitled particle simulation project

A simple Python 3 simulation of particles trapped by current-carrying wires.

Released under GNU 3 License

Written by Cameron McGarry

## Install

1. Pull files to local repository
2. (Recommended) Create and activate virtual environment
3. `pip install -r requirements`
4. `cc -fPIC -shared -o USP/wire_segment_fn.so USP/wire_segment_fn.c`
5. Enjoy!

## Test

Run `test.py`, or for coverage use `sh run_coverage.sh`
