
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

PetscErrorCode set_scatter(mat_Ctx *user);

PetscErrorCode set_scatter(mat_Ctx *user)
{

  Vec x;
  VecCreate(PETSC_COMM_WORLD,&x);
  VecSetSizes(x,PETSC_DECIDE,2*user->n);
  VecSetFromOptions(x);
  VecScatterCreate(user->x1,NULL,x, user->isg[0], &user->scatter_to_1);
  VecScatterCreate(user->x2,NULL,x, user->isg[1], &user->scatter_to_2); 
  VecDestroy(&x); 
  return(0);
}