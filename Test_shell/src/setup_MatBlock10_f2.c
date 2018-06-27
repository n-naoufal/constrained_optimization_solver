  
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


PetscErrorCode setup_MatBlock10_f2(mat_Ctx *user);

PetscErrorCode setup_MatBlock10_f2(mat_Ctx *user)
{

  char             mat[PETSC_MAX_PATH_LEN];
  PetscBool        flg_matrix;
  PetscViewer      view;

  PetscOptionsGetString(NULL,NULL,"-matF",mat,PETSC_MAX_PATH_LEN,&flg_matrix);
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,mat,FILE_MODE_READ,&view);
  MatCreate(PETSC_COMM_WORLD,&user->subA[1]);
  MatLoad(user->subA[1],view); 
  PetscViewerDestroy(&view); 
  
  MatGetSize(user->subA[1],&user->m,NULL);

  return(0);
}