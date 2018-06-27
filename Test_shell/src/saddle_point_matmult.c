
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

PetscErrorCode saddle_point_matmult(Mat A,Vec x,Vec y);

PetscErrorCode saddle_point_matmult(Mat A,Vec x,Vec y)
{
  void           *ptr;

  MatShellGetContext(A,&ptr);
  mat_Ctx         *user = (mat_Ctx *)ptr;

/*Load values of  x1, x2 from x : use SCATTER_REVERSE*/

  VecScatterBegin(user->scatter_to_1,x,user->x1,INSERT_VALUES,SCATTER_REVERSE);
  VecScatterEnd(user->scatter_to_1,x,user->x1,INSERT_VALUES,SCATTER_REVERSE);

  VecScatterBegin(user->scatter_to_2,x,user->x2,INSERT_VALUES,SCATTER_REVERSE);
  VecScatterEnd(user->scatter_to_2,x,user->x2,INSERT_VALUES,SCATTER_REVERSE);

  MatMult(user->subA[0], user->x1, user->xtmp);
  MatMultAdd(user->subA[1],user->x2,user->xtmp,user->y1);

  MatMult(user->subA[1], user->x1, user->ytmp);
  MatMultAdd(user->subA[2],user->x2,user->ytmp,user->y2);

/*Load result to y : this time, use SCATTER_FORWARD*/
  VecScatterBegin(user->scatter_to_1,user->y1,y,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(user->scatter_to_1,user->y1,y,INSERT_VALUES,SCATTER_FORWARD);

  VecScatterBegin(user->scatter_to_2,user->y2,y,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(user->scatter_to_2,user->y2,y,INSERT_VALUES,SCATTER_FORWARD);


  return(0);
}