These *.mat files contain the data used in the analysis and discussion of the journal article 'numerical considerations in the simulation of equatorial spread F.'
Variable Naming Convention:
1. circAdvecData.mat
n0          - initial density distribution in the circular advection test;
n_scheme    - density distribution at t=1 using the specified advection scheme;
x and y     - coordinate.


2. numDensity.mat
nSeg1_Seg2_Seg3 - simulated electron number density.    
    'Seg1' represents grid size:
    1   - 101*201 grids;
    2   - 201*401 grids;
    3   - 301*601 grids;
    4   - 401*801 grids.
    'Seg2' represents the advection scheme used:
    upwind1     - 1st-order upwind scheme;
    vanLeer2    - 2nd-order vanLeer scheme;
    02/04/06/08 - Corresponding alpha value for the 7th-order PDM scheme.
    'Seg3' represents the simulation time in seconds.


