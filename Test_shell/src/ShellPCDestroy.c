
#include <petscmat.h>
#include <petscksp.h>
#include <petsc.h>

/* ------------------------------------------------------------------- */
/*
   ShellPCDestroy - This routine destroys a user-defined
   preconditioner context.

   Input Parameter:
.  pc_Ctx - user-defined preconditioner context
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



PetscErrorCode ShellPCDestroy(PC pc);

PetscErrorCode ShellPCDestroy(PC pc) 
{

  pc_Ctx        *userpc;

  PCShellGetContext(pc,(void**)&userpc);
  PCDestroy(&userpc->pc);
  PCDestroy(&userpc->pc1);
  MatDestroy(&userpc->BD);
  VecDestroy(&userpc->diag);

  return 0;
}


