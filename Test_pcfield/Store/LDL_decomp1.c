
#include <petscmat.h>
/*
   This file extract the factor D from LDL decomposition of ZPrZ
    Input : 
             d              : The diagonal of D^-1
            sd           : The square diagonal of D^-1
            chKr            : The cholesky decomposition of the norm matrix Kr
            n_sensors       : The number of experimental sensors

    Output : Matrix 
*/
#undef __FUNCT__
#define __FUNCT__ "LDL_decomp1"
void LDL_decomp1(Vec *d, Vec *sd, Mat chKr, PetscInt n_sensors);


void LDL_decomp1(Vec *d, Vec *sd, Mat chKr, PetscInt n_sensors)
{ 

  Mat               chKr_T; 


/*    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Extract D factor from Z'PrZ = LDL'
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */


  // d = diagonal(D)   ( = sD^2 where sD the main diagonal of chKr)
  MatTranspose(chKr,MAT_INITIAL_MATRIX,&chKr_T);
  MatGetDiagonal(chKr_T,*sd);
  VecPointwiseMult(*d, *sd,*sd);

  // dinv = sqrt(diagonal(D^-1))
  VecReciprocal(*sd); 

  MatDestroy(&chKr_T);

  return;
}