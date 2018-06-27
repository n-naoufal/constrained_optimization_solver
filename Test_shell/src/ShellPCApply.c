
#include <petscmat.h>
#include <petscksp.h>
#include <petsc.h>
/* ------------------------------------------------------------------- */
/*
   ShellPCApply - This routine demonstrates the use of a
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


PetscErrorCode ShellPCApply(PC pc,Vec x,Vec y);
PetscErrorCode ShellPCApply(PC pc,Vec x,Vec y)
{
  Mat           pmat;
  void          *ptr;
  pc_Ctx        *userpc;


  PCShellGetContext(pc,(void**)&userpc);
  PCGetOperators(pc,NULL,&pmat);
  MatShellGetContext(pmat,&ptr);
  mat_Ctx        *user = (mat_Ctx*)ptr;
  Vec   xtmp, xtmp1;


/*Load values of  x1, x2 from x : use SCATTER_REVERSE*/

  VecScatterBegin(user->scatter_to_1,x,user->x1,INSERT_VALUES,SCATTER_REVERSE);
  VecScatterEnd(user->scatter_to_1,x,user->x1,INSERT_VALUES,SCATTER_REVERSE);

  VecScatterBegin(user->scatter_to_2,x,user->x2,INSERT_VALUES,SCATTER_REVERSE);
  VecScatterEnd(user->scatter_to_2,x,user->x2,INSERT_VALUES,SCATTER_REVERSE);

  VecDuplicate(user->x1, &xtmp);
  VecDuplicate(xtmp, &xtmp1);

/*   Apply Augmented Lgrangian preconditioner :
---------------------------------------------------------------------------------
(y1) = ( D   Bz )^{-1} (x1) = (-sqrt(D)^{-1}  -D^{-1}*Bz ) (        sqrt(D)^{-1}              0        ) (x1)
(y2)   ( Bz  Tz )      (x2)   ( 0                   I    ) ( -Eu^{-1}El^{-1}*Bz*D^{-1}  Eu^{-1}El^{-1} ) (x2)

where E = ZTZ-ZBZ D^-1 ZBZ.  note also that D = -sqrt(D)^{-1}*sqrt(D)^{-1}= -abs(D) so that E is SPD

   i.e.  y1 = D^-1 * x1 - D^-1*Bz * y2
         y2 = Eu^{-1}El^{-1}* (-Bz*D^{-1} x1 + x2) 
---------------------------------------------------------------------------------
*/

/*   xtmp = D^-1 * x1      */

  VecPointwiseMult(xtmp, userpc->diag,user->x1);

/*   xtmp1 = -Bz*D^{-1} x1 + x2   */

  MatMult(userpc->BD,user->x1,xtmp1);
  VecSet(xtmp1,-1.0);
  VecAXPY(xtmp1,1.0,user->x2);

/*  y2 = Eu^{-1}El^{-1}* xtmp1   */

  PCApply(userpc->pc, xtmp1, user->y2);
  VecDestroy(&xtmp1);
/* y1 = - D^-1*Bz * y2 */ 

  MatMultTranspose(userpc->BD,user->y2,user->y1);
  VecScale(user->y1, -1.0);

/*  y1 = xtmp + y1     */
  VecAXPY(user->y1,1.0,xtmp);
  VecDestroy(&xtmp);


/*Load result to y : this time, use SCATTER_FORWARD*/
  VecScatterBegin(user->scatter_to_1,user->y1,y,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(user->scatter_to_1,user->y1,y,INSERT_VALUES,SCATTER_FORWARD);

  VecScatterBegin(user->scatter_to_2,user->y2,y,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(user->scatter_to_2,user->y2,y,INSERT_VALUES,SCATTER_FORWARD);

  return 0;
}