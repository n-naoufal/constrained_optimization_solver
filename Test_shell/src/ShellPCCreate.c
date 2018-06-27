

#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>
#include <petscsys.h>

/*
   ShellPCCreate - This routine creates a user-defined
   preconditioner context.

   Output Parameter:
.  shell - user-defined preconditioner context
*/

typedef struct {
    PC         pc, pc1;
	Mat        BD;
	Vec        diag;
} pc_Ctx;


PetscErrorCode ShellPCCreate(pc_Ctx**);
PetscErrorCode ShellPCCreate(pc_Ctx **userpc)
{
	pc_Ctx  *userpc1;
	
	PetscNew(&userpc1);
	*userpc       = userpc1;
	return 0;
}