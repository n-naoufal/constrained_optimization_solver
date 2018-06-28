# Designing solvers for constrained optimization quadratic problems


## Instructions for users
* Step 1.  Download Petsc, see https://www.mcs.anl.gov/petsc/download/index.html
* Step 2.  Configure Petsc, invoke the following commands from the top level PETSc directory: 
```
./config/configure.py  --prefix=$PWD/Install --download-superlu --download-superlu_dist --download-fblaslapack --download-scalapack --download-mumps --download-mumps-serial --download-hypre --download-parmetis --download-metis --download-mpich --download-parms --download-pastix --download-ptscotch  --download-suitesparse --with-debugging=1 --with-X=1
```
or on a server using available packages:
```
./config/configure.py  --prefix=$PWD/Install --download-fblaslapack=/packages/fblaslapack-3.4.2.tar.gz --download-scalapack=/packages/scalapack-2.0.2.tgz --download-mumps=/packages/MUMPS_5.0.2-p2.tar.gz --download-hypre=/packages/hypre-2.10.0b-p4.tar.gz --download-metis=/packages/metis-5.1.0.tar.gz  --download-mpich=/packages/mpich-3.1.3.tar.gz --download-parms=/packages/pARMS_3.2p4.tar.gz --download-pastix=/packages/pastix_5.2.3.tar.bz2 --download-ptscotch=/packages/scotch_6.0.4-p1.tar.gz  --download-suitesparse=/packages/SuiteSparse-4.4.3.tar.gz --with-debugging=1 --with-X=1
```
* Step 3.  build and install Petsc, invoke the following commands one after the other: 
```
make PETSC_DIR=/petsc PETSC_ARCH=arch-linux2-c-debug all
make PETSC_DIR=/petsc PETSC_ARCH=arch-linux2-c-debug install
make PETSC_DIR=/petsc/Install PETSC_ARCH="" test
make PETSC_DIR=/petsc/Install PETSC_ARCH= streams
```
## Instructions for developers

* If you want to rebuild Petsc, from the top level PETSc directory invoke the following commands: 
```
cd petsc/
export PETSC_ARCH=arch-linux2-c-debug
./$PETSC_ARCH/lib/petsc/conf/reconfigure-$PETSC_ARCH.py  (--with-debugging=Note  2/3 times faster,  =yes)
make PETSC_DIR=/petsc PETSC_ARCH=arch-linux2-c-debug all
make PETSC_DIR=/petsc PETSC_ARCH=arch-linux2-c-debug install
```
-----------------------------------------------------------------------------------------
* In order to create a KSP method, for example a new conjugate gradient method (mycg): 
```
- Create folder mycg in /petsc/src/ksp/ksp/impls/cg/
- In this folder, create your own method mycg.c and a makefile
- in /petsc/src/ksp/ksp/impls/cg/ edit the makefile
- In petsc/src/ksp/ksp/interface/itregis.c add : 
	PETSC_EXTERN PetscErrorCode KSPCreate_MYCG(KSP); 
	KSPMYCG and references to KSPCreate_MYCG
- In petsc/include/petscksp.h add 
	#define KSPMYCG         "mycg"
  ```
