static char help[] = "\n";




#include <petscksp.h>

typedef struct {
  PetscBool userPC, userKSP; /* user defined preconditioner and matrix for the Schur complement */
  Mat       A;       /* block matrix */
  PetscInt   n;
  Mat       subA[4]; /* the four blocks */
  Mat       myS;     /* the approximation of the Schur complement */
  Vec       x, b; /* solution, rhs and temporary vector */
  IS        isg[2];  /* index sets of split "0" and "1" */
} Field;

PetscErrorCode SetupMatBlock00(Field*);  /* setup the block 00 */
PetscErrorCode SetupMatBlock01(Field*);  /* setup the block 01 */
PetscErrorCode SetupMatBlock10(Field*);  /* setup the block 10 */
PetscErrorCode SetupMatBlock11(Field*);  /* setup the block 11 */
PetscErrorCode Rhs(Field*);                                               /* rhs vector */
PetscErrorCode SetupApproxSchur(Field*);  /* approximation of the Schur complement */

//PetscErrorCode WriteSolution(Field*); /* write solution to file */



PetscErrorCode SetupPC(Field *s, KSP ksp)
{

  PC             pc;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = PCFieldSplitSetIS(pc, "0", s->isg[0]);CHKERRQ(ierr);
  ierr = PCFieldSplitSetIS(pc, "1", s->isg[1]);CHKERRQ(ierr);
  if (s->userPC) {
    ierr = PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_USER, s->myS);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

// PetscErrorCode WriteSolution(Field *s)
// {
//   PetscMPIInt       size;
//   PetscInt          n,i,j;
//   const PetscScalar *array;
//   PetscErrorCode    ierr;

//   PetscFunctionBeginUser;
//   /* write data (*warning* only works sequential) */
//   MPI_Comm_size(MPI_COMM_WORLD,&size);
//   /*ierr = PetscPrintf(PETSC_COMM_WORLD," number of processors = %D\n",size);CHKERRQ(ierr);*/
//   if (size == 1) {
//     PetscViewer viewer;
//     ierr = VecGetArrayRead(s->x, &array);CHKERRQ(ierr);
//     ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "solution.dat", &viewer);CHKERRQ(ierr);
//     ierr = PetscViewerASCIIPrintf(viewer, "# x, y, u, v, p\n");CHKERRQ(ierr);
//     for (j = 0; j < s->ny; j++) {
//       for (i = 0; i < s->nx; i++) {
//         n    = j*s->nx+i;
//         ierr = PetscViewerASCIIPrintf(viewer, "%.12g %.12g %.12g %.12g %.12g\n", (double)(i*s->hx+s->hx/2),(double)(j*s->hy+s->hy/2), (double)PetscRealPart(array[n]), (double)PetscRealPart(array[n+s->nx*s->ny]),(double)PetscRealPart(array[n+2*s->nx*s->ny]));CHKERRQ(ierr);
//       }
//     }
//     ierr = VecRestoreArrayRead(s->x, &array);CHKERRQ(ierr);
//     ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//   }
//   PetscFunctionReturn(0);
// }

PetscErrorCode SetupIndexSets(Field *s)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* the two index sets */
  ierr = MatNestGetISs(s->A, s->isg, NULL);CHKERRQ(ierr);
  /*  ISView(isg[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  /*  ISView(isg[1],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}

PetscErrorCode SetupVectors(Field *s)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* solution vector x */
  ierr = MatCreateVecs(s->A,&s->x,NULL);CHKERRQ(ierr);
  // ierr = VecCreate(PETSC_COMM_WORLD, &s->x);CHKERRQ(ierr);
  // ierr = VecSetSizes(s->x, PETSC_DECIDE, 2*s->n);CHKERRQ(ierr);
  // ierr = VecSetType(s->x, VECMPI);CHKERRQ(ierr);
  
  /*  ierr = VecSetRandom(s->x, NULL);CHKERRQ(ierr); */
  /*  ierr = VecView(s->x, (PetscViewer) PETSC_VIEWER_DEFAULT);CHKERRQ(ierr); */

  /* exact solution y */
//  ierr = VecDuplicate(s->x, &s->y);CHKERRQ(ierr);
//  ierr = StokesExactSolution(s);CHKERRQ(ierr);
  /*  ierr = VecView(s->y, (PetscViewer) PETSC_VIEWER_DEFAULT);CHKERRQ(ierr); */

  /* rhs vector b */
  ierr = VecDuplicate(s->x, &s->b);CHKERRQ(ierr);
  ierr = Rhs(s);CHKERRQ(ierr);
  /*ierr = VecView(s->b, (PetscViewer) PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);*/
  PetscFunctionReturn(0);
}


PetscErrorCode Rhs(Field *s)
{
  char             vec[PETSC_MAX_PATH_LEN];
  PetscBool        flg_vec=PETSC_FALSE;
  Vec              b0, b1;
  PetscViewer      view;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;


  ierr = VecGetSubVector(s->b,s->isg[0],&b0);CHKERRQ(ierr);
  ierr = VecSet(b0,0.0);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(s->b,s->isg[0],&b0);CHKERRQ(ierr);
  ierr = VecGetSubVector(s->b,s->isg[1],&b1);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-vecrhs",vec,PETSC_MAX_PATH_LEN,&flg_vec);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vec,FILE_MODE_READ,&view);CHKERRQ(ierr);
  ierr = VecLoad(b1,view); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view); CHKERRQ(ierr);
  ierr = VecRestoreSubVector(s->b,s->isg[1],&b1);CHKERRQ(ierr);
//  ierr = VecView(s->b,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}





PetscErrorCode SetupMatBlock00(Field *s)
{
  char             mat[PETSC_MAX_PATH_LEN];
  PetscBool        flg_matrix=PETSC_FALSE;
  PetscViewer      view;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  /* A[0] is N-by-N */

  ierr = PetscOptionsGetString(NULL,NULL,"-matZAZ",mat,PETSC_MAX_PATH_LEN,&flg_matrix);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,mat,FILE_MODE_READ,&view);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&s->subA[0]);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(s->subA[0], "a00_");CHKERRQ(ierr);
// Consider MPI in parallel setting
//  ierr = MatSetType(s->subA[0],MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatLoad(s->subA[0],view); CHKERRQ(ierr);
  ierr = MatGetSize(s->subA[0],&s->n,NULL);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view); CHKERRQ(ierr);

// For alternator test case add this value on the element (1,1) because too small 
//  ierr = MatSetValue(s->subA[0],0,0,-8.12978e+09,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(s->subA[0],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(s->subA[0],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return(0);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupMatBlock01(Field *s)
{
  char             mat[PETSC_MAX_PATH_LEN];
  PetscBool        flg_matrix=PETSC_FALSE;
  PetscViewer      view;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  /* A[1] is N-by-N */

  ierr = PetscOptionsGetString(NULL,NULL,"-matZBZ",mat,PETSC_MAX_PATH_LEN,&flg_matrix);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,mat,FILE_MODE_READ,&view);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&s->subA[1]);CHKERRQ(ierr);
// Consider MPI in parallel setting
//  ierr = MatSetType(s->subA[1],MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatLoad(s->subA[1],view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(s->subA[1], "a01_");CHKERRQ(ierr);
  ierr = MatAssemblyBegin(s->subA[1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(s->subA[1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return(0);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupMatBlock10(Field *s)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* A[2] is minus transpose of A[1] */
  s->subA[2] = s->subA[1];
  ierr = MatSetOptionsPrefix(s->subA[2], "a10_");CHKERRQ(ierr);
  ierr = MatAssemblyBegin(s->subA[2],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(s->subA[2],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupMatBlock11(Field *s)
{
  char             mat[PETSC_MAX_PATH_LEN];
  PetscBool        flg_matrix=PETSC_FALSE;
  PetscViewer      view;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  /* A[3] is N-by-N null matrix */
  
  ierr = PetscOptionsGetString(NULL,NULL,"-matZPrZ",mat,PETSC_MAX_PATH_LEN,&flg_matrix);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,mat,FILE_MODE_READ,&view);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&s->subA[3]);CHKERRQ(ierr);
// Consider MPI in parallel setting
//  ierr = MatSetType(s->subA[3],MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatLoad(s->subA[3],view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(s->subA[3], "a11_");CHKERRQ(ierr);
  ierr = MatAssemblyBegin(s->subA[3], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(s->subA[3], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupApproxSchur(Field *s)
{

  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Schur complement approximation: myS = A11 - A10 diag(A00)^(-1) A01 */
  /* note: A11 is zero */
  /* note: in real life this matrix would be build directly, */
  /* i.e. without MatMatMult */

  /* compute: A11 - A10 diag(A00)^(-1) A01 */
  /* restore A10 */

  // ierr = MatConvert(s->subA[0], MATSAME,MAT_INITIAL_MATRIX,&s->myS);CHKERRQ(ierr);
  // ierr = MatAXPY(s->myS,1.0,s->subA[3],DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  // Approximation of the Shur complement by a sparse matrix : Tz-Az (see PhD manuscript)
  MatConvert(s->subA[0], MATSAME,MAT_INITIAL_MATRIX,&s->myS);
  ierr = MatScale(s->myS,-1.0);CHKERRQ(ierr);
  ierr = MatAXPY(s->myS,1.0,s->subA[3],DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode SetupMatrix(Field *s)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = SetupMatBlock00(s);CHKERRQ(ierr);
  ierr = SetupMatBlock01(s);CHKERRQ(ierr);
  ierr = SetupMatBlock10(s);CHKERRQ(ierr);
  ierr = SetupMatBlock11(s);CHKERRQ(ierr);
  ierr = MatCreateNest(PETSC_COMM_WORLD, 2, NULL, 2, NULL, s->subA, &s->A);CHKERRQ(ierr);
  ierr = SetupApproxSchur(s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




int main(int argc, char **argv)
{
  Field          s;
  KSP            ksp;
  PetscErrorCode ierr;

  ierr     = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  s.userPC = PETSC_FALSE;
  ierr     = PetscOptionsHasName(NULL,NULL, "-user_pc", &s.userPC);CHKERRQ(ierr);

  ierr = SetupMatrix(&s);CHKERRQ(ierr);
  ierr = SetupIndexSets(&s);CHKERRQ(ierr);
  ierr = SetupVectors(&s);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, s.A, s.A);CHKERRQ(ierr);

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = SetupPC(&s, ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, s.b, s.x);CHKERRQ(ierr);
  
//  ierr = WriteSolution(&s);CHKERRQ(ierr);
//  ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);



  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&s.subA[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&s.subA[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&s.subA[2]);CHKERRQ(ierr);
  ierr = MatDestroy(&s.subA[3]);CHKERRQ(ierr);
//  ierr = MatDestroy(&s.A);CHKERRQ(ierr);
  ierr = VecDestroy(&s.x);CHKERRQ(ierr);
  ierr = VecDestroy(&s.b);CHKERRQ(ierr);
  ierr = MatDestroy(&s.myS);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
