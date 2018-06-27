
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

Mat read_file(FILE *file,PetscBool flag_sym, int flag_read, char mat[PETSC_MAX_PATH_LEN]);
PetscErrorCode set_saddle_rhs(mat_Ctx *user);

PetscErrorCode set_saddle_rhs(mat_Ctx *user)
{
  Mat               Z;
  Vec               b1, bz;
  PetscReal         val;
  long int          dummy;
  PetscInt          i, n_phys;
  FILE              *rhs_file, *Z_file;
  PetscBool         flg_rhs=PETSC_FALSE;
  char              rhs[PETSC_MAX_PATH_LEN], matZ[PETSC_MAX_PATH_LEN];

  Z = read_file(Z_file,PETSC_FALSE, 4, matZ);
  MatGetSize(Z,&n_phys,NULL);
  /*Construction of the rhs  */
  PetscOptionsGetString(NULL,NULL,"-rhs",rhs,PETSC_MAX_PATH_LEN,&flg_rhs);
  if (flg_rhs){
  VecCreate(PETSC_COMM_WORLD,&b1);
  VecSetSizes(b1,PETSC_DECIDE,n_phys);
  VecSetFromOptions(b1);

  PetscFOpen(PETSC_COMM_WORLD,rhs,"r",&rhs_file);
  for (i=0; i<n_phys; i++) {
    fscanf(rhs_file,"%ld %le\n",&dummy,(double*)&val);
    VecSetValues(b1,1,&i,&val,INSERT_VALUES);
  }
  VecAssemblyBegin(b1);
  VecAssemblyEnd(b1);
  fflush(stdout);
  fclose(rhs_file);
  }
  VecGetSubVector(user->b,user->isg[1],&bz);
  MatMultTranspose(Z,b1,bz);
  MatDestroy(&Z);
  VecDestroy(&b1);
  VecRestoreSubVector(user->b,user->isg[1],&bz);

  return(0);
}