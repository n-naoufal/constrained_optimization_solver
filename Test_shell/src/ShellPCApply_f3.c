
#include <petscmat.h>
#include <petscksp.h>
#include <petsc.h>

/* ------------------------------------------------------------------- */
/*
   ShellPCApply_f3 - This routine demonstrates the use of a
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


PetscErrorCode ShellPCApply_f3(PC pc,Vec x,Vec y);
PetscErrorCode ShellPCApply_f3(PC pc,Vec x,Vec y)
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
  VecDuplicate(user->x2, &xtmp1);

/*   Apply shur preconditioner :
---------------------------------------------------------------------------------
 
   (y1) = ( G  F^T)^{-1} (x1) = ( G^-1 0) ( I -F^T) ( G         0      ) (I  0) ( G^-1 0) (x1)
   (y2)   ( F  0  )      (x2)   ( 0    I) ( 0    I) ( 0  -(FG^-1F^T)^-1) (-F I) ( 0    I) (x2)
   
   i.e.  y1 = G^-1 (x1+F^T*S^-1(-FG^-1*x1+x2))
         y2 = -S^-1(-FG^-1*x1+x2)

  S =  (FG^-1F^T)^-1 (application through solver)             
---------------------------------------------------------------------------------
*/

/*  xtmp =  G^-1*x1      */

  VecPointwiseMult(xtmp, userpc->diag,user->x1);

/*   xtmp1 = -FG^-1*x1+x2      */

  MatMult(user->subA[1],xtmp,xtmp1);
  VecDestroy(&xtmp); 
  VecSet(xtmp1,-1.0);
  VecAXPY(xtmp1,1.0,user->x2);
 
/*   y2 = -S^-1*xtmp1  (Apply the inner preconditionner pc) */

  PCApply(userpc->pc,xtmp1, user->y2);
  VecSet(user->y2,-1.0); 
  VecDestroy(&xtmp1); 

/*  xtmp1 = x1+F^T*y2    */
  VecDuplicate(user->x1, &xtmp1);
  MatMultTranspose(user->subA[1], user->y2, xtmp1);
  VecSet(xtmp1,-1.0); 
  VecAXPY(xtmp1,1.0,user->x1);

/*  y1 =  G^-1*xtmp1      */
  VecPointwiseMult(user->y1, userpc->diag,xtmp1);
  VecDestroy(&xtmp1); 
/*Load result to y : this time, use SCATTER_FORWARD*/
  VecScatterBegin(user->scatter_to_1,user->y1,y,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(user->scatter_to_1,user->y1,y,INSERT_VALUES,SCATTER_FORWARD);

  VecScatterBegin(user->scatter_to_2,user->y2,y,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(user->scatter_to_2,user->y2,y,INSERT_VALUES,SCATTER_FORWARD);

  return 0;
}