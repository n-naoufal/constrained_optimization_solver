
#include <petscmat.h>
/*
    This file construct the constraint block  F = [Z'BZ  L]
    Input : 
            Z'BZ            : The projection of the matrix B onto the nullspace of C
            L               : Factor of the followinf decomposition Z'PrZ= LDL^T
            n_sensors       : The number of experimental sensors

    Output : Matrix 
*/
#undef __FUNCT__
#define __FUNCT__ "set_constraint_block"
Mat set_constraint_block(Mat ZBZ, Mat L, PetscInt n_sensors);

Mat set_constraint_block(Mat ZBZ, Mat L, PetscInt n_sensors)
{ 

  Mat               F; 
  PetscReal         val, dtol=1.e-16;
  PetscInt          i, j, nz_F, n_F, nnz_L,n_ZBZ, nnz_ZBZ;
  PetscInt          ncols_ZBZ, ncols_L, k_L;
  const PetscInt    *cols_ZBZ, *cols_L;
  const PetscScalar *vals_ZBZ, *vals_L;
  MatInfo           info_ZBZ, info_L;



  /* Matrices information */


  MatGetInfo(ZBZ,MAT_LOCAL,&info_ZBZ);
  nnz_ZBZ = info_ZBZ.nz_allocated;
  MatGetSize(ZBZ,NULL,&n_ZBZ);



  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
               construct the block F = [ZBZ  L]
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  // preallocatin F
  MatCreate(PETSC_COMM_WORLD,&F);
  n_F=n_ZBZ+n_sensors;
  MatSetSizes(F,PETSC_DECIDE,PETSC_DECIDE,n_ZBZ,n_F);
  MatSetFromOptions(F);
  MatGetInfo(L,MAT_LOCAL,&info_L);
  nnz_L = info_L.nz_allocated;
  nz_F=nnz_ZBZ+nnz_L;
  MatSeqAIJSetPreallocation(F,nz_F/n_F,NULL);
  MatSetOption(F,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);

  // Fill F entries
  for (i=0; i<n_ZBZ; i++) {
    MatGetRow(ZBZ,i,&ncols_ZBZ,&cols_ZBZ,&vals_ZBZ);
    for (j=0; j<ncols_ZBZ; j++) {
      val = PetscAbsScalar(vals_ZBZ[j]);
      if (val > dtol) {
        MatSetValues(F,1,&i,1,&cols_ZBZ[j],&vals_ZBZ[j],INSERT_VALUES);
      }
    }
    MatRestoreRow(ZBZ,i,&ncols_ZBZ,&cols_ZBZ,&vals_ZBZ);

    MatGetRow(L,i,&ncols_L,&cols_L,&vals_L);
    for (j=0; j<ncols_L; j++) {
      val = PetscAbsScalar(vals_L[j]);
      if (val > dtol) {
        k_L=cols_L[j]+n_ZBZ;
        MatSetValues(F,1,&i,1,&k_L,&vals_L[j],INSERT_VALUES);
      }
    }
    MatRestoreRow(L,i,&ncols_L,&cols_L,&vals_L);
  }

  MatAssemblyBegin(F,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(F,MAT_FINAL_ASSEMBLY);

  return F;
}