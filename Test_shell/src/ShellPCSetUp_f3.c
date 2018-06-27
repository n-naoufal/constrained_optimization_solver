
/* ------------------------------------------------------------------- */
/*
   ShellPCSetUp_f3 - This routine sets up a user-defined
   preconditioner context.

   Input Parameters:
.  pc    - preconditioner object
.  pmat  - preconditioner matrix
.  x     - vector

   Output Parameter:
.  pc_Ctx - fully set up user-defined preconditioner context

*/
#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>


typedef struct {
  IS         isg[2];  /* Index set of 1st and 2nd physical degrees of freedom */   
  PetscInt   n,m;  /*  size of each block */
  Mat        subA[3]; /* the four blocks ( ZBZ=ZBZ' ) */
  Vec        x1, x2, xtmp;
  Vec        y1, y2, ytmp;
  Vec        b;
  VecScatter scatter_to_1, scatter_to_2;
} mat_Ctx;

typedef struct {
  PC         pc, pc1;
  Mat        BD;
  Vec        diag;
} pc_Ctx;

PetscErrorCode ShellPCSetUp_f3(PC pc,Mat pmat,Vec x);

PetscErrorCode ShellPCSetUp_f3(PC pc,Mat pmat,Vec x)
{
  pc_Ctx        *userpc;
  
  void          *ptr;

  Mat            F, E, product;
  PetscReal      gamma=0.5;
  
  PCShellGetContext(pc,(void**)&userpc);
  MatShellGetContext(pmat,&ptr);

  mat_Ctx        *user = (mat_Ctx*)ptr;
  
/*  G approximates H   */
  

/*  -----> uncomment if you choose  G = diag(H)   */
/*----------------------------------------------------------------------*/
  Vec            diag;
  VecCreate(PETSC_COMM_WORLD,&diag);
  VecSetSizes(diag,PETSC_DECIDE,user->n);
  VecSetFromOptions(diag);
  MatGetDiagonal(user->subA[0],diag);
  VecReciprocal(diag);
  /* Stoke diag in the PCshell context */
  userpc->diag = diag;
  // VecView(diag,  PETSC_VIEWER_STDOUT_SELF);
  Mat            D;
  MatCreate(PETSC_COMM_WORLD,&D);
  MatSetSizes(D,PETSC_DECIDE,PETSC_DECIDE,user->n,user->n);
  MatSetFromOptions(D);
  MatSeqAIJSetPreallocation(D,1,NULL);
  MatSetOption(D,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);
  MatDiagonalSet(D,diag,INSERT_VALUES);
/*----------------------------------------------------------------------*/

 /* E = F*G^-1*F^T */ 
  MatMatMult(user->subA[1],D,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&product);
  MatDestroy(&D);
  MatMatTransposeMult(product,user->subA[1],MAT_INITIAL_MATRIX, 2.6,&E);
  MatDestroy(&product);

  /* Declare E as SPD matrix to use Cholesky or ICC*/
  MatSetOption(E,MAT_SPD,PETSC_TRUE);
  
/* Define preconditionner pcphy, based on the incomplete/ complete cholesky factorization of E */
  PCCreate(PETSC_COMM_WORLD,&userpc->pc);
  PCSetOperators(userpc->pc, E, E);

  PCSetType(userpc->pc, PCCHOLESKY);
  PCFactorSetMatSolverPackage(userpc->pc,MATSOLVERMUMPS);
  PCFactorSetUpMatSolverPackage(userpc->pc);
  PCFactorGetMatrix(userpc->pc,&F);

/* ICNTL(7) (sequential matrix ordering): 5 (METIS) */
  MatMumpsSetIcntl(F,7,5);

/*  ICNTL(22) (in-core/out-of-core facility): 0/1 */
  MatMumpsSetIcntl(F,22,1);

/*  ICNTL(24) (detection of null pivot rows): 1   */
  MatMumpsSetIcntl(F,24,1);

/*  ICNTL(14) (percentage increase in the estimated working space) */
  MatMumpsSetIcntl(F,14,50);

/*  CNTL(3) (absolute pivoting threshold):      1e-06   */
  MatMumpsSetCntl(F,3,1.e-6);
  PCSetUp(userpc->pc);
  MatDestroy(&E);


  return 0;
}