static char help[] = " \n\
\n\n";                                                                                                    
/*
Application of Block preconditionners on the saddle point problem in the following :

------------------------------------------------------------------------------------------------
----> 1st form  where  Z is a projection onto the nullspace of C                                   
                                                                                                     
| Z^TAZ     Z^TBZ | |x_Z1|      |  0  |              | Z    0 | |x_Z1|                               
|                 | |    |  =   |     |    where x = |        | |    |                               
| Z^TBZ     Z^TPrZ| |x_Z2|      |Z^T*f|              | 0    Z | |x_Z2|                               
                                                                                                     
 For this form we take the following preconditioner : 
 
 ( D   Bz )^{-1} = (-sqrt(D)^{-1}  -D^{-1}*Bz ) (        sqrt(D)^{-1}              0        ) 
 ( Bz  Tz )        ( 0                   I    ) ( -Eu^{-1}El^{-1}*Bz*D^{-1}  Eu^{-1}El^{-1} ) 

 where E = ZTZ - ZBZ D^-1 ZBZ 
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------

----> 2nd form  where    Z^TPrZ = L D L^T   and F = | Z^TBZ L |                                                                                                     
                                                                                                     
| Z^TAZ      0       Z^TBZ  |  |    -x_Z1   |      |    0   |                                        
|                           |  |            |      |        |         | H   F^T| | t  |      |  0  | 
|  0         D^-1      L^T  |  | -DL^T x_Z2 |  =   |    0   |  <===>  |        | |    |  =   |     | 
|                           |  |            |      |        |         | F    0 | |x_Z2|      |Z^T*f| 
| Z^TBZ      L       0      |  |    x_Z2    |      | -Z^T*f |                                        

 For this form we take the following preconditioner  : 

1. -------------> Ref. (PhD thesis, Sylvain Mercier p. 113) -->  
New block triangular preconditioners for saddle point linear systems with highly singular (1, 1) blocks. 
T. Z. Huang, G. H. ChenG, and L. Li. Mathematical Problems in Engineering, 13, 2009.


 ( LL^T  2 F^T     )^{-1} = ( L^{-T} L^{-1}  0      ) ( I  2*gamma*F^T ) 
 ( 0    -1/gamma*I )        ( 0             -gamma*I) ( 0  I           )        

 E = H + gamma* F^T F  = LL^T     

 2.-------------> Shur complement decomposition 

 ( G  F^T)^{-1} =   ( G^-1 0) ( I -F^T) ( G         0      ) (I  0) ( G^-1 0)                                                         
 ( F  0  )          ( 0    I) ( 0    I) ( 0  -(FG^-1F^T)^-1) (-F I) ( 0    I)      

 Where G approximates H                                                   
-------------------------------------------------------------------------------------------------


                                                                                                     */
#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>
#include <petscsys.h>

typedef struct {
  IS         isg[2];  /* Index set of 1st and 2nd physical degrees of freedom */   
  PetscInt   n, m;  /*  size of each block */
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


/* form 1 subroutines */
PetscErrorCode ShellPCApply(PC pc,Vec x,Vec y);
PetscErrorCode ShellPCDestroy(PC pc);
PetscErrorCode ShellPCSetUp(PC pc,Mat pmat,Vec x);
PetscErrorCode ShellPCCreate(pc_Ctx**);
PetscErrorCode free_saddle_point_context(mat_Ctx *user);
PetscErrorCode saddle_point_matmult(Mat A,Vec x,Vec y);
PetscErrorCode set_is(mat_Ctx *user);
PetscErrorCode set_saddle_rhs(mat_Ctx *user);
PetscErrorCode set_scatter(mat_Ctx *user);
PetscErrorCode set_workspace(mat_Ctx *user);
PetscErrorCode setup_MatBlock00(mat_Ctx *user);
PetscErrorCode setup_MatBlock10(mat_Ctx *user);
PetscErrorCode setup_MatBlock11(mat_Ctx *user);
Mat read_file(FILE *file,PetscBool flag_sym, int flag_read, char mat[PETSC_MAX_PATH_LEN]);

/* form 2 subroutines */
PetscErrorCode setup_MatBlock00_f2(mat_Ctx *user);
PetscErrorCode setup_MatBlock10_f2(mat_Ctx *user);
PetscErrorCode set_is_f2(mat_Ctx *user);
PetscErrorCode set_workspace_f2(mat_Ctx *user);
PetscErrorCode set_scatter_f2(mat_Ctx *user);
PetscErrorCode set_saddle_rhs_f2(mat_Ctx *user);
PetscErrorCode saddle_point_matmult_f2(Mat A,Vec x,Vec y);
PetscErrorCode ShellPCSetUp_f2(PC pc,Mat pmat,Vec x);
PetscErrorCode ShellPCApply_f2(PC pc,Vec x,Vec y);
PetscErrorCode ShellPCSetUp_f3(PC pc,Mat pmat,Vec x);
PetscErrorCode ShellPCApply_f3(PC pc,Vec x,Vec y);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{

 Mat            A;
 Vec            x;    /* approx solution, RHS, exact solution */
 mat_Ctx        user;
 pc_Ctx         *userpc;
 PetscBool      flg_form1=PETSC_FALSE, flg_form2=PETSC_FALSE, flg_lagr=PETSC_FALSE, flg_shur=PETSC_FALSE;;
 char              mat[PETSC_MAX_PATH_LEN];
 KSP            ksp;
 PC             pc;
 PetscErrorCode ierr;



 PetscInitialize(&argc,&args,(char*)0,help);
 ierr = PetscMemorySetGetMaximumUsage();CHKERRQ(ierr);

 ierr = PetscOptionsGetString(NULL,NULL,"-form1",mat,PETSC_MAX_PATH_LEN,&flg_form1);CHKERRQ(ierr);
 ierr = PetscOptionsGetString(NULL,NULL,"-form2",mat,PETSC_MAX_PATH_LEN,&flg_form2);CHKERRQ(ierr);

/*    --------------------------------------------------------------------------------------
                    Form 1     [ZAZ ZBZ; ZBZ ZTZ] [x1;x2] = [0 ; Z'f2]      
      --------------------------------------------------------------------------------------    */

 if (flg_form1) {

/* Build different part of the matrix context */

 setup_MatBlock00(&user);  /* Block (1,1) of the sadle point matrix */
 setup_MatBlock10(&user);  /* Block (2,1) = (1,2) of the sadle point matrix */
 setup_MatBlock11(&user);  /* Block (2,2) of the sadle point matrix */
 set_is(&user);            /* Index of the sadle point matrix */
 set_workspace(&user);     /* Workspace of the matrix context */
 set_scatter(&user);       /* To scatter vectors */

/* Creation of the MATSHELL */
   ierr = MatCreateShell(PETSC_COMM_WORLD,2*user.n,2*user.n,2*user.n,2*user.n,(void*)&user,&A);CHKERRQ(ierr);
   ierr = MatShellSetOperation(A,MATOP_MULT,(void(*)(void))saddle_point_matmult);CHKERRQ(ierr);

 /*  Construction of the rhs */
   set_saddle_rhs(&user);


/*    --------------------------------------------------------------------------------------
                    Form 2     [H F^T; F 0] [x1;x2] = [0 ; -Z'f2]      
      --------------------------------------------------------------------------------------    */

 } else if (flg_form2) {

/* Build different part of the matrix context */

 setup_MatBlock00_f2(&user);  /* Block (1,1) of the sadle point matrix */
 setup_MatBlock10_f2(&user);  /* Block (2,1) of the sadle point matrix */
 set_is_f2(&user);            /* Index of the sadle point matrix */
 set_workspace_f2(&user);     /* Workspace of the matrix context */
 set_scatter_f2(&user);       /* To scatter vectors */

/* Creation of the MATSHELL */
   ierr = MatCreateShell(PETSC_COMM_WORLD,user.n+user.m,user.n+user.m,user.n+user.m,user.n+user.m,(void*)&user,&A);CHKERRQ(ierr);
   ierr = MatShellSetOperation(A,MATOP_MULT,(void(*)(void))saddle_point_matmult_f2);CHKERRQ(ierr);

 /*  Construction of the rhs */
   set_saddle_rhs_f2(&user);

 } else {

  SETERRQ(PETSC_COMM_WORLD,1,"Must choose form1 or form2 from runtime options to continue ! ");
}

/*    --------------------------------------------------------------------------------------
                                  APPLICATION OF THE SOLVER     
      --------------------------------------------------------------------------------------    */

 /* Solution vector   */
ierr = VecDuplicate(user.b,&x);

 /*   Create linear solver context*/

ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
ierr = KSPSetOperators(ksp, A, A);CHKERRQ(ierr);

/* Extracting the KSP and PC contexts from the KSP context */
ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
ierr = PCSetType(pc,PCSHELL);CHKERRQ(ierr);
ierr = KSPSetType(ksp, KSPGMRES);CHKERRQ(ierr);


 /* SetUp PC shell*/

ShellPCCreate(&userpc);

if (flg_form1) {

  /* PCShell for form 1 */
  ierr = PCShellSetApply(pc,ShellPCApply);CHKERRQ(ierr);
  ierr = PCShellSetContext(pc,userpc);CHKERRQ(ierr); /* Let it here in this order otherwise you get memory Segmentation error*/
  ShellPCSetUp(pc,A,x);

} else if (flg_form2) {  
    /* PCShell for form 2 */
 ierr = PetscOptionsGetString(NULL,NULL,"-lagrangian",mat,PETSC_MAX_PATH_LEN,&flg_lagr);CHKERRQ(ierr);
 ierr = PetscOptionsGetString(NULL,NULL,"-shur",mat,PETSC_MAX_PATH_LEN,&flg_shur);CHKERRQ(ierr);

 if (flg_lagr) {
    /* Lagragian preconditoner */ 
  ierr = PCShellSetApply(pc,ShellPCApply_f2);CHKERRQ(ierr);
  ierr = PCShellSetContext(pc,userpc);CHKERRQ(ierr); /* Let it here in this order otherwise you get memory Segmentation error*/
  ShellPCSetUp_f2(pc,A,x);
 } else if (flg_shur) {
    /* Lagragian preconditoner */ 
  ierr = PCShellSetApply(pc,ShellPCApply_f3);CHKERRQ(ierr);
  ierr = PCShellSetContext(pc,userpc);CHKERRQ(ierr); /* Let it here in this order otherwise you get memory Segmentation error*/
  ShellPCSetUp_f3(pc,A,x);
 }  else {
  SETERRQ(PETSC_COMM_WORLD,1,"Must choose a preconditioner (lagrangian, shur, ...) from runtime options to continue ! ");
 }
}


ierr = PCShellSetDestroy(pc, ShellPCDestroy);CHKERRQ(ierr);
ierr = PCShellSetName(pc,"Block Preconditionner");CHKERRQ(ierr);

/*
  Set runtime options, e.g., -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
  These options will override those specified above as long as KSPSetFromOptions() is called 
  _after_ any other customization routines.
*/
ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

 /* Solve */ 
ierr = KSPSolve(ksp, user.b, x);CHKERRQ(ierr);

KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);

 /* Clean context */
free_saddle_point_context(&user);

ierr = PetscFinalize();
return 0;
}