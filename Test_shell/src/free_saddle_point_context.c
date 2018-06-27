
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

PetscErrorCode free_saddle_point_context(mat_Ctx *user);

PetscErrorCode free_saddle_point_context(mat_Ctx *user)
{

  ISDestroy(&user->isg[0]);
  ISDestroy(&user->isg[1]);
  MatDestroy(&user->subA[0]);
  MatDestroy(&user->subA[1]);
  MatDestroy(&user->subA[2]);
  VecDestroy(&user->x1);
  VecDestroy(&user->x2);
  VecDestroy(&user->xtmp);
  VecDestroy(&user->ytmp);
  VecDestroy(&user->y1);
  VecDestroy(&user->y2);
  VecDestroy(&user->b);
  VecScatterDestroy(&user->scatter_to_1);
  VecScatterDestroy(&user->scatter_to_2);
  return(0);
}