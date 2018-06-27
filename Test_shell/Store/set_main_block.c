
#include <petscmat.h>
/*
    This file construct the main block  H = [Z'AZ  D^-1]
    Input : 
            Z'AZ            : The projection of the matrix A onto the nullspace of C
            sd              :   sqrt(diagonal(D^-1))

    Output : Matrix 
*/
#undef __FUNCT__
#define __FUNCT__ "set_main_block"
Mat set_main_block(Mat ZAZ, Vec sd, PetscInt n_sensors);


Mat set_main_block(Mat ZAZ, Vec sd, PetscInt n_sensors)
{ 

  Mat               H, DDinv; 
  Vec               dinv;
  PetscReal         val, dtol=1.e-16;
  PetscInt          i, j, m_H, nz_H, n_ZAZ, nnz_ZAZ;
  PetscInt          ncols_DDinv, ncols_ZAZ, k_DD, i_DD;
  const PetscInt    *cols_DDinv, *cols_ZAZ;
  const PetscScalar *vals_DDinv, *vals_ZAZ;
  MatInfo           info_ZAZ;




  /* Matrices information */

  MatGetInfo(ZAZ,MAT_LOCAL,&info_ZAZ);
  nnz_ZAZ = info_ZAZ.nz_allocated;
  MatGetSize(ZAZ,NULL,&n_ZAZ);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
               construct the block H = [Z'AZ  0 ; 0  D^-1]
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*     Build  D^-1             */
  VecDuplicate(sd,&dinv);
  VecPointwiseMult(dinv, sd,sd);
  MatCreate(PETSC_COMM_WORLD,&DDinv);
  MatSetSizes(DDinv,PETSC_DECIDE,PETSC_DECIDE,n_sensors,n_sensors);
  MatSetFromOptions(DDinv);
  MatSeqAIJSetPreallocation(DDinv,1,NULL);
  MatSetOption(DDinv,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);
  MatDiagonalSet(DDinv,dinv,INSERT_VALUES);


  // preallocatin H
  MatCreate(PETSC_COMM_WORLD,&H);
  m_H  = n_ZAZ+n_sensors;
  MatSetSizes(H,PETSC_DECIDE,PETSC_DECIDE,m_H,m_H);
  MatSetFromOptions(H);
  nz_H = nnz_ZAZ+n_sensors;
  MatSeqAIJSetPreallocation(H,nz_H/m_H,NULL);
  MatSetOption(H,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);

  MatScale(ZAZ, -1.0);

  // Fill out H  with Z'AZ

  for (i=0; i<n_ZAZ; i++) {
    MatGetRow(ZAZ,i,&ncols_ZAZ,&cols_ZAZ,&vals_ZAZ);
    for (j=0; j<ncols_ZAZ; j++) {
      val = PetscAbsScalar(vals_ZAZ[j]);
      if (val > dtol) {
        MatSetValues(H,1,&i,1,&cols_ZAZ[j],&vals_ZAZ[j],INSERT_VALUES);
      }
    }
    MatRestoreRow(ZAZ,i,&ncols_ZAZ,&cols_ZAZ,&vals_ZAZ);
  }
  //  Fill out H  with D^-1
  for (i=0; i<n_sensors; i++) {
    MatGetRow(DDinv,i,&ncols_DDinv,&cols_DDinv,&vals_DDinv);
    for (j=0; j<ncols_DDinv; j++) {
      val = PetscAbsScalar(vals_DDinv[j]);
      if (val > dtol) {
        i_DD=i+n_ZAZ;
        k_DD=cols_DDinv[j]+n_ZAZ;
        MatSetValues(H,1,&i_DD,1,&k_DD,&vals_DDinv[j],INSERT_VALUES);
      }
    }
    MatRestoreRow(DDinv,i,&ncols_DDinv,&cols_DDinv,&vals_DDinv);
  }

  MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);

  

  MatDestroy(&DDinv);
  VecDestroy(&dinv);


  

  return H;
}