
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

PetscErrorCode set_workspace_f2(mat_Ctx *user);

PetscErrorCode set_workspace_f2(mat_Ctx *user)
{

  MatCreateVecs(user->subA[0],&user->x1,&user->y1);
  VecDuplicate(user->x1,&user->xtmp);

  VecCreate(PETSC_COMM_WORLD,&user->x2);
  VecSetSizes(user->x2,PETSC_DECIDE,user->m);
  VecSetFromOptions(user->x2);
  VecDuplicate(user->x2,&user->y2);
  VecDuplicate(user->x2,&user->ytmp);

  VecCreate(PETSC_COMM_WORLD,&user->b);
  VecSetSizes(user->b,PETSC_DECIDE,user->n+user->m);
  VecSetFromOptions(user->b);

  return(0);
}