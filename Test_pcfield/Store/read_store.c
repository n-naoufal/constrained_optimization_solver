
#include <petscmat.h>
/*
    This file read ASCII files
    Output : Matrix 
*/
#undef __FUNCT__
#define __FUNCT__ "read_store"
Mat read_store(FILE *file,PetscBool flag_sym, int flag_read, char mat[PETSC_MAX_PATH_LEN]);


Mat read_store(FILE *file,PetscBool flag_sym, int flag_read, char mat[PETSC_MAX_PATH_LEN])
{ 
  Mat            M;
  PetscInt       shift=1,nsizes,i;
  PetscInt       m,n,nz;
  PetscInt       col,row,sizes[3];
  PetscScalar    val;
  PetscBool      flg, flg_matrix;
  

  if (flag_read==1) {
    PetscOptionsGetString(NULL,NULL,"-matM",mat,PETSC_MAX_PATH_LEN,&flg_matrix);
  } else if (flag_read==2) {
    PetscOptionsGetString(NULL,NULL,"-matK",mat,PETSC_MAX_PATH_LEN,&flg_matrix);
  } else if (flag_read==3) {
    PetscOptionsGetString(NULL,NULL,"-matprod",mat,PETSC_MAX_PATH_LEN,&flg_matrix);
  } else if (flag_read==4) {
    PetscOptionsGetString(NULL,NULL,"-matZ",mat,PETSC_MAX_PATH_LEN,&flg_matrix);
  } else if (flag_read==5) {
    PetscOptionsGetString(NULL,NULL,"-matobs",mat,PETSC_MAX_PATH_LEN,&flg_matrix);
  } else if (flag_read==6) {
    PetscOptionsGetString(NULL,NULL,"-matKr",mat,PETSC_MAX_PATH_LEN,&flg_matrix);
  } else if (flag_read==7) {
    PetscOptionsGetString(NULL,NULL,"-matchKr",mat,PETSC_MAX_PATH_LEN,&flg_matrix);
  } 

  PetscOptionsHasName(NULL,NULL,"-noshift",&flg);
  if (flg) shift = 0;
  if (flg_matrix) {

    PetscFOpen(PETSC_COMM_WORLD,mat,"r",&file);
    nsizes = 3;
    PetscOptionsGetIntArray(NULL,NULL,"-nosizesinfile",sizes,&nsizes,&flg);
    if (flg) {
      m  = sizes[0];
      n  = sizes[1];
      nz = 2*(sizes[2]-n)+n;
    } else {
      fscanf(file,"%d %d %d\n",&m,&n,&nz);
    }
    if (flag_sym) {
      PetscInt nnz;
      nnz =2*(nz-n)+n;
      PetscPrintf(PETSC_COMM_WORLD,"m: %ld, n: %ld, nz: %ld \n", m,n,nnz);
      MatCreate(PETSC_COMM_WORLD,&M);
      MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,m,n);
      MatSetFromOptions(M);
      MatSetUp(M);
      MatSetType(M,MATSEQAIJ);
      MatSetOption(M,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);
      MatSeqAIJSetPreallocation(M,nz/m,NULL);
      MatSetOption(M,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);

    } else {
      PetscPrintf(PETSC_COMM_WORLD,"m: %ld, n: %ld, nz: %ld \n", m,n,nz);
      MatCreate(PETSC_COMM_WORLD,&M);
      MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,m,n);
      MatSetFromOptions(M);
      MatSetUp(M);
      MatSetType(M,MATSEQAIJ);
      MatSetOption(M,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);
      MatSeqAIJSetPreallocation(M,nz/m,NULL);
      MatSetOption(M,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);
    }  
    for (i=0; i<nz; i++) {
      fscanf(file,"%d %d %le\n",&row,&col,(double*)&val);
      row -= shift; col -= shift;  /* set index set starts at 0 */
      if (val>1e-16){
      MatSetValues(M,1,&row,1,&col,&val,INSERT_VALUES);
    }
    }
    MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);
    fflush(stdout);
    fclose(file);
    
  } 
  return M;
}