
#include <petscmat.h>
#include <petscksp.h>
#include <petsc.h>

/* ------------------------------------------------------------------- */
/*
   ShellPCApply_f2 - This routine demonstrates the use of a
   user-provided preconditioner.

   Input Parameters:
+  pc - preconditioner object
-  x - input vector

   Output Parameter:
.  y - preconditioned vector

*/

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


PetscErrorCode ShellPCApply_f2(PC pc,Vec x,Vec y);
PetscErrorCode ShellPCApply_f2(PC pc,Vec x,Vec y)
{
  Mat           pmat;
  void          *ptr;
  pc_Ctx        *userpc;
  PCShellGetContext(pc,(void**)&userpc);
  PCGetOperators(pc,NULL,&pmat);
  MatShellGetContext(pmat,&ptr);

  mat_Ctx         *user = (mat_Ctx*)ptr;
  Vec              xtmp, xtmp1;
  PetscReal        gamma=0.5, beta;


/*Load values of  x1, x2 from x : use SCATTER_REVERSE*/

  VecScatterBegin(user->scatter_to_1,x,user->x1,INSERT_VALUES,SCATTER_REVERSE);
  VecScatterEnd(user->scatter_to_1,x,user->x1,INSERT_VALUES,SCATTER_REVERSE);

  VecScatterBegin(user->scatter_to_2,x,user->x2,INSERT_VALUES,SCATTER_REVERSE);
  VecScatterEnd(user->scatter_to_2,x,user->x2,INSERT_VALUES,SCATTER_REVERSE);

  VecDuplicate(user->x1, &xtmp);
  VecDuplicate(xtmp, &xtmp1);

/*   Apply Augmented Lgrangian preconditioner :
---------------------------------------------------------------------------------
   PhD thesis Sylvain p. 113
   E = H + gamma* F^T F  = LL^T
   
   (y1) = ( LL^T  2 F^T     )^{-1} (x1) = ( L^{-T} L^{-1}  0      ) ( I  2*gamma*F^T ) (x1)
   (y2)   ( 0    -1/gamma*I )      (x2)   ( 0             -gamma*I) ( 0  I           ) (x2)
   
   i.e.  y1 = L^{-T} L^{-1} (x1 + 2*gamma*F^T*x2)
         y2 = -gamma*x2 

---------------------------------------------------------------------------------
*/ 
/*    y2 = -gamma * x2  */
  
  VecCopy(user->x2, user->y2);
  beta = - gamma;
  VecScale(user->y2,beta);

/*   xtmp = F^T x2       */

  MatMultTranspose(user->subA[1], user->x2, xtmp);

/*   x1 = x1 + (2*gamma)* xtmp   */

  beta=2.0*gamma;
  VecAXPY( user->x1, beta, xtmp);

/*   Apply the inner preconditionner pc   */
  PCApply(userpc->pc,user->x1, user->y1);


/*Load result to y : this time, use SCATTER_FORWARD*/
  VecScatterBegin(user->scatter_to_1,user->y1,y,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(user->scatter_to_1,user->y1,y,INSERT_VALUES,SCATTER_FORWARD);

  VecScatterBegin(user->scatter_to_2,user->y2,y,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(user->scatter_to_2,user->y2,y,INSERT_VALUES,SCATTER_FORWARD);

  return 0;
}