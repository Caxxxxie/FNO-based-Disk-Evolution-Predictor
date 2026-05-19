# FARGO3D Vendor Notes

This is a local copy of the FARGO3D source tree from:

```text
https://github.com/FARGO3D/fargo3d
```

We use it to generate small disk time-series datasets for the time-dependent
PPDONet project. Build artifacts and simulation outputs are ignored by git.

Useful local commands:

```bash
cd fargo3d
make SETUP=fargo_nu PARALLEL=0 GPU=0
./fargo3d ../fargo_data/smoke_fargo_nu/par/case_000.par
```

For larger runs, rebuild with MPI/GPU options outside the small sanity-check
workflow and keep raw outputs out of the repository.
