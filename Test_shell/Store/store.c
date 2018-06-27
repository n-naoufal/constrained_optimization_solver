static char help[] = "Store projected matrices";                                                                                                    

#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>
#include <petscsys.h>

Mat read_store(FILE *file,PetscBool flag_sym, int flag_read, char mat[PETSC_MAX_PATH_LEN]);
void LDL_decomp1(Vec *d, Vec *sd, Mat chKr, PetscInt n_sensors);
Mat LDL_decomp2(Mat obs, Mat Z , Mat chKr, Vec sdinv, PetscInt n_sensors, PetscScalar alpha);
Mat set_main_block(Mat ZAZ, Vec sd, PetscInt n_sensors);
Mat set_constraint_block(Mat ZBZ, Mat L, PetscInt n_sensors);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            K,prod,Z; 
  Mat            Ks, Bs, prods;
  Mat            B, ZAZ, ZBZ, ZPrZ;
  Mat            ZB, ZK, Zprod;
  Vec            v;
  char           matM[PETSC_MAX_PATH_LEN], matK[PETSC_MAX_PATH_LEN];
  char           matprod[PETSC_MAX_PATH_LEN],matZ[PETSC_MAX_PATH_LEN], mat[PETSC_MAX_PATH_LEN];

  PetscScalar    om2=100.0;// 1.59155E+02*2*M_PI*1.59155E+02*2*M_PI;
  FILE           *M_file, *K_file, *prod_file;
  FILE           *Z_file;
  PetscBool      flg_form1, flg_form2;
  PetscViewer      view;
  PetscErrorCode ierr;


  PetscInitialize(&argc,&args,(char*)0,help);
  ierr = PetscMemorySetGetMaximumUsage();CHKERRQ(ierr);

/*----------------------------------------------------------------------------------------------------
    READ MATRICES K, M, PROD and Z AND CONSTRUCT THEIR PROJECTION ONTO THE NULLSPACE OF C 
 ---------------------------------------------------------------------------------------------------*/

  printf("\n");
  printf("\n ------->  Reading matrices from ASCII files ");
  printf("\n");

/* Read K, M, Z  matrices */

  ierr   = PetscPrintf(PETSC_COMM_WORLD,"\n *** The Mass Matrix M ***\n");CHKERRQ(ierr);
  B = read_store(M_file,PETSC_TRUE, 1, matM);
  ierr   = PetscPrintf(PETSC_COMM_WORLD,"\n *** The nullspace basis Z ***\n");CHKERRQ(ierr);
  Z = read_store(Z_file,PETSC_FALSE, 4, matZ);
  ierr   = PetscPrintf(PETSC_COMM_WORLD,"\n *** The Stiffness Matrix K ***\n");CHKERRQ(ierr);
  K = read_store(K_file,PETSC_TRUE, 2, matK);

/* Construct B = K-om^2*M and ZBZ  then free B space*/
  ierr = MatCreateVecs(B,NULL,&v);CHKERRQ(ierr);
  ierr = MatTranspose(B,MAT_INITIAL_MATRIX,&Bs);CHKERRQ(ierr);
  ierr = MatAXPY(B,1.0,Bs,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDestroy(&Bs);CHKERRQ(ierr);
  ierr = MatGetDiagonal(B,v);CHKERRQ(ierr);
  ierr = VecScale(v, 0.5);CHKERRQ(ierr);
  ierr = MatDiagonalSet(B,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  
  ierr =  MatCreateVecs(K,&v,NULL);CHKERRQ(ierr);
  ierr = MatTranspose(K,MAT_INITIAL_MATRIX,&Ks);CHKERRQ(ierr);
  ierr = MatAXPY(K,1.0,Ks,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDestroy(&Ks);CHKERRQ(ierr);
  ierr = MatGetDiagonal(K,v);CHKERRQ(ierr);
  ierr = VecScale(v, 0.5);CHKERRQ(ierr);
  ierr = MatDiagonalSet(K,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);

  ierr = MatAYPX(B,-om2,K,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  ierr = MatTransposeMatMult(Z,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&ZB);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = MatMatMult(ZB,Z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&ZBZ);CHKERRQ(ierr);
  ierr = MatDestroy(&ZB);CHKERRQ(ierr);CHKERRQ(ierr);

/* Construct ZAZ=Z(-K)Z and free K matrix */
  ierr = MatScale(K, -1.0);CHKERRQ(ierr);
 //  ierr = MatPtAP(K,Z, MAT_INITIAL_MATRIX,PETSC_DEFAULT,&ZAZ);CHKERRQ(ierr);
  ierr = MatTransposeMatMult(Z,K,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&ZK);CHKERRQ(ierr);
  ierr = MatDestroy(&K);CHKERRQ(ierr);
  ierr = MatMatMult(ZK,Z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&ZAZ);CHKERRQ(ierr);
  ierr = MatDestroy(&ZK);CHKERRQ(ierr);CHKERRQ(ierr);

 /* Read prod matrix */
  ierr   = PetscPrintf(PETSC_COMM_WORLD,"\n *** The block matrix (2,2) ***\n");CHKERRQ(ierr);
  prod = read_store(prod_file,PETSC_TRUE, 3, matprod);

 /* Construct ZPrZ and free prod matrix */
  ierr = MatCreateVecs(prod,&v,NULL);CHKERRQ(ierr);
  ierr = MatTranspose(prod,MAT_INITIAL_MATRIX,&prods);CHKERRQ(ierr);
  ierr = MatAXPY(prod,1.0,prods,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDestroy(&prods);CHKERRQ(ierr);
  
  ierr = MatGetDiagonal(prod,v);CHKERRQ(ierr);
  ierr = VecScale(v, 0.5);CHKERRQ(ierr);
 // VecView(v,PETSC_VIEWER_STDOUT_SELF);
 // ierr = MatDiagonalSet(prod,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);


// //  ierr = MatPtAP(prod,Z, MAT_INITIAL_MATRIX,PETSC_DEFAULT,&ZPrZ);CHKERRQ(ierr);
  ierr = MatTransposeMatMult(Z,prod,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Zprod);CHKERRQ(ierr);
  ierr = MatDestroy(&prod);CHKERRQ(ierr);
  ierr = MatMatMult(Zprod,Z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&ZPrZ);CHKERRQ(ierr);
  ierr = MatDestroy(&Zprod);CHKERRQ(ierr);CHKERRQ(ierr);
  
  ierr = PetscOptionsGetString(NULL,NULL,"-form1",mat,PETSC_MAX_PATH_LEN,&flg_form1);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-form2",mat,PETSC_MAX_PATH_LEN,&flg_form2);CHKERRQ(ierr);

  if (flg_form1) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n ------->   Write matrices in Form 1 [ZAZ ZBZ; ZBZ ZTZ]  ...\n");
/*----------------------------------------------------------------------------------------------------
                       STORE MATRICES ZAZ, ZBZ, ZPrZ in Binary format 
 ---------------------------------------------------------------------------------------------------*/
    ierr = MatDestroy(&Z);CHKERRQ(ierr);
/* Save  ZAZ and free prod matrix */

    PetscBool flg_matrix =PETSC_FALSE;
    ierr = PetscOptionsGetString(NULL,NULL,"-matZAZ",mat,PETSC_MAX_PATH_LEN,&flg_matrix);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Write matrix ZAZ in binary to 'ZAZ.dat' ...\n");CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,mat,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
    ierr = MatView(ZAZ,view);CHKERRQ(ierr);
    ierr = MatDestroy(&ZAZ);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);

/* Save  ZBZ and free prod matrix */
    flg_matrix =PETSC_FALSE;
    ierr = PetscOptionsGetString(NULL,NULL,"-matZBZ",mat,PETSC_MAX_PATH_LEN,&flg_matrix);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Write matrix ZBZ in binary to 'ZBZ.dat' ...\n");CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,mat,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
    ierr = MatView(ZBZ,view);CHKERRQ(ierr);
    ierr = MatDestroy(&ZBZ);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);

/* Save  ZPrZ and free prod matrix */
    flg_matrix =PETSC_FALSE;
    ierr = PetscOptionsGetString(NULL,NULL,"-matZPrZ",mat,PETSC_MAX_PATH_LEN,&flg_matrix);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Write matrix ZPrZ in binary to 'ZPrZ.dat' ...\n");CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,mat,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
    ierr = MatView(ZPrZ,view);CHKERRQ(ierr);
    ierr = MatDestroy(&ZPrZ);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);


  } else if (flg_form2) {
    ierr = MatDestroy(&ZPrZ);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n ------->   Write matrices in Form 1 [H F'; F  0]  ...\n");

/*----------------------------------------------------------------------------------------------------
                       STORE MATRICES H and F in Binary format 
 ---------------------------------------------------------------------------------------------------*/
    FILE           *obs_file, *chKr_file;
    char           matobs[PETSC_MAX_PATH_LEN], matchKr[PETSC_MAX_PATH_LEN];
    PetscInt       n_sensors;
    Mat            L, H, F, obs, chKr;
    Vec            d, sdinv;
    PetscScalar    alpha=0.5;

    ierr   = PetscPrintf(PETSC_COMM_WORLD,"\n *** The observation matrix ***\n");CHKERRQ(ierr);
    obs = read_store(obs_file,PETSC_FALSE, 5, matobs); 
    ierr   = PetscPrintf(PETSC_COMM_WORLD,"\n *** The upper triangle matrix chKr=cholesky(Kr) ***\n");CHKERRQ(ierr);
    chKr = read_store(chKr_file,PETSC_FALSE, 7, matchKr);
    ierr = MatGetSize(chKr,NULL,&n_sensors);CHKERRQ(ierr);

    ierr = VecCreateSeq(PETSC_COMM_WORLD,n_sensors,&d);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(d);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(d);CHKERRQ(ierr);
    ierr = VecDuplicate(d,&sdinv);CHKERRQ(ierr);
    LDL_decomp1(&d, &sdinv, chKr, n_sensors);
    L = LDL_decomp2(obs, Z , chKr, sdinv, n_sensors, alpha);
    ierr = MatDestroy(&Z);CHKERRQ(ierr);
/* Destroying unecessary data, Free work space */ 
    ierr = MatDestroy(&obs);CHKERRQ(ierr);
    ierr = MatDestroy(&chKr);CHKERRQ(ierr);
    H = set_main_block(ZAZ, sdinv, n_sensors);
    F = set_constraint_block(ZBZ, L, n_sensors);
/* Destroying unecessary data, Free work space */ 
    ierr = MatDestroy(&ZAZ);CHKERRQ(ierr);
    ierr = MatDestroy(&ZBZ);CHKERRQ(ierr);
    ierr = VecDestroy(&d);CHKERRQ(ierr);
    ierr = VecDestroy(&sdinv);CHKERRQ(ierr);

/* Save  H and free prod matrix */
    PetscBool flg_matrix =PETSC_FALSE;
    ierr = PetscOptionsGetString(NULL,NULL,"-matH",mat,PETSC_MAX_PATH_LEN,&flg_matrix);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Write matrix H in binary to 'H.dat' ...\n");CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,mat,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
    ierr = MatView(H,view);CHKERRQ(ierr);
    ierr = MatDestroy(&H);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);


/* Save  F and free prod matrix */
    flg_matrix =PETSC_FALSE;
    ierr = PetscOptionsGetString(NULL,NULL,"-matF",mat,PETSC_MAX_PATH_LEN,&flg_matrix);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Write matrix F in binary to 'F.dat' ...\n");CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,mat,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
    ierr = MatView(F,view);CHKERRQ(ierr);
    ierr = MatDestroy(&F);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);

  } else {
    
    SETERRQ(PETSC_COMM_WORLD,1,"Must choose form1 or form2 from runtime options to continue ! ");
  }
  printf("\n");
  printf("\n ------->   Matrices saved ! ");
  printf("\n");

  ierr = PetscFinalize();
  return 0;
}

