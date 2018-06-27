
#include <petscvec.h>
#include <petscmat.h>

typedef struct {
  IS         isg[2];  /* Index set of 1st and 2nd physical degrees of freedom */   
  PetscInt   n,m;  /*  size of each block */
  Mat        subA[3]; /* the four blocks ( ZBZ=ZBZ' ) */
  Vec        x1, x2, xtmp;
  Vec        y1, y2, ytmp;
  Vec        b;
  VecScatter scatter_to_1, scatter_to_2;
} mat_Ctx;

PetscErrorCode set_is(mat_Ctx *user);


PetscErrorCode set_is(mat_Ctx *user)
{
 PetscInt n, i;
 n = user->n;
 PetscInt index1[n];
 PetscInt index2[n];
 for (i = 0; i < n; ++i)
 {
  index1[i] = i;
  index2[i] = i+n;
}

ISCreateGeneral(PETSC_COMM_WORLD, n, index1, PETSC_COPY_VALUES, &user->isg[0]);
ISCreateGeneral(PETSC_COMM_WORLD, n, index2, PETSC_COPY_VALUES, &user->isg[1]);

return(0);
}