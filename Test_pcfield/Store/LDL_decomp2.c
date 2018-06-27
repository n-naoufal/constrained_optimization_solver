
#include <petscmat.h>
/*
    This file extract the factor L from LDL decomposition of ZPrZ
    Input : obs             : The observation matrix
            Z               : the nullspace basis of C
            chKr            : The cholesky decomposition of the norm matrix Kr
            sdinv           : The square diagonal of D^-1
            n_sensors       : The number of experimental sensors
            alpha           : A weighting coefficient 

    Output : Matrix 
*/
#undef __FUNCT__
#define __FUNCT__ "LDL_decomp2"
Mat LDL_decomp2(Mat obs, Mat Z , Mat chKr, Vec sdinv, PetscInt n_sensors, PetscScalar alpha);
Mat LDL_decomp2(Mat obs, Mat Z , Mat chKr, Vec sdinv, PetscInt n_sensors, PetscScalar alpha)
{ 

  Mat               chKr_T, J, ObsJ, Z_T, L; 
  Mat               Sinv;
  PetscScalar       c_alpha;




/*    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Extract L factor from Z'PrZ = LDL'
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */


  /* ------------------------------------------------------------------------ /
  / get L from Z'PrZ = Z' *(Obs*Kr*Obs')* Z = c_alpha Z'*Obs*(J*D*J')*Obs'*Z  /
  /                 = c_alpha (Z'*Obs*J) * D * (J*Obs'*Z)                     /
  /                 = L * D * L'                                              /
  / -------------------------------------------------------------------------*/                                                                           

   // matrix J  ( = J_ch * S^-1 )
 
  MatCreate(PETSC_COMM_WORLD,&Sinv);
  MatSetSizes(Sinv,PETSC_DECIDE,PETSC_DECIDE,n_sensors,n_sensors);
  MatSetFromOptions(Sinv);
  MatSeqAIJSetPreallocation(Sinv,1,NULL);
  MatSetOption(Sinv,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);
  MatDiagonalSet(Sinv,sdinv,INSERT_VALUES);
  MatTranspose(chKr,MAT_INITIAL_MATRIX,&chKr_T);
  MatMatMult(chKr_T,Sinv,MAT_INITIAL_MATRIX, PETSC_DEFAULT,&J);


   // matrix L  ( = sqrt(c_alpha) *(Z'*Obs*J))
  MatMatMult(obs,J,MAT_INITIAL_MATRIX, PETSC_DEFAULT,&ObsJ);
  MatTranspose(Z,MAT_INITIAL_MATRIX,&Z_T);

  MatMatMult(Z_T,ObsJ,MAT_INITIAL_MATRIX, PETSC_DEFAULT,&L);
  c_alpha=-2.0*alpha/(1-alpha);
  MatScale(L, sqrt(abs(c_alpha)));

  MatDestroy(&chKr_T);
  MatDestroy(&J);
  MatDestroy(&ObsJ);
  MatDestroy(&Z_T);
  MatDestroy(&Sinv);
  return L;
}