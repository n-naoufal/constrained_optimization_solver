
PETSC_DIR        = /petsc-3.7.6/Install
PETSC_ARCH       =  arch-linux2-c-debug
LOCDIR           = /petsc-3.7.6/Install
EXAMPLESC        = test.c 
DATAFILESPATH    = /petsc-3.7.6/Install/Test_shell
MANSEC           = KSP
CLEANFILES       = rhs.vtk solution.vtk
NP               = 1

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

#--------------------------------------------------------------------------------------------
#                 §§§ READ THESE INSTRUCTIONS FIRST §§§                                     #
#--------------------------------------------------------------------------------------------
#                                                                                           #
# 1) First Build the different blocks of the saddle point matrix from the set of testcases. # 
#   to do so :- choose the form to use $(OPTForm1)  or $(OPTForm2) in target "run_store"    #
#             - choose the testcase from  DATAFILESPATH1 (test0, test1, test2, ...)         # 
#             - DATASTORE should point toward the same testcase number (test0, test1, ...)  # 
#             - then, ---> make store --------> make run_store                              #     
#                                                                                           #        
# 2) Test                                                                                   #         
#    choose the form to use $(OPTForm1)  or $(OPTForm2) in target "case"                    #     
#    then -------> make main -----------> make case                                         #     
#    You can edit the solver runtime options in OPTGNR                                      #      
# -------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------
#               Store matrices in Form 1 (ZAZ, ZBZ and ZPrZ ) or Form 2  (H and F)          #
#--------------------------------------------------------------------------------------------

# Compilation 
csrc = $(wildcard Store/*.c) 
obj = $(csrc: .c=.o)
store: $(obj) 
	-${CLINKER} -o  $@ $^ ${PETSC_KSP_LIB}

# Where to take the testcase 
DATAFILESPATH1    = ~/petsc-3.7.6/Install/Test_pcfield/matrices/test1
# Where to store the saddle point matrix blocks 
DATASTORE   = ~/petsc-3.7.6/Install/Test_pcfield/matrices/test1
# Needed files to construct the saddle point matrix blocks
DATAFILESstore =  -matM $(DATAFILESPATH1)/matM -matK $(DATAFILESPATH1)/matK -matprod $(DATAFILESPATH1)/matprod -matZ $(DATAFILESPATH1)/matZ -matobs $(DATAFILESPATH1)/matobs  -matchKr $(DATAFILESPATH1)/matchKr

# Choose the saddle point problem in form 1 or form 2
OPTForm1 = -form1 -matZAZ $(DATASTORE)/ZAZ.dat -matZBZ $(DATASTORE)/ZBZ.dat -matZPrZ $(DATASTORE)/ZPrZ.dat
OPTForm2 = -form2 -matH $(DATASTORE)/H.dat  -matF $(DATASTORE)/F.dat

# Execution of the storing command 
run_store: 
		-@${MPIEXEC}  -n 1 ./store  $(DATAFILESstore) $(OPTForm1) > result1.tmp 2>&1;


#--------------------------------------------------------------------------------------------
#                            Applying solution method                                       #
#--------------------------------------------------------------------------------------------



# Compilation 
csrc = $(wildcard *.c) 
obj = $(csrc: .c=.o)
main: $(obj) 
	-${CLINKER} -o  $@ $^ ${PETSC_KSP_LIB}

#  Solver runtime options  ! note that if you chose an other type of solver, this will override the block resolution
# ---------------------------------------------------------------------------------------------------------------------------------

# OUTER LOOP
KSPA = -ksp_type gmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full 
KSPB  = -ksp_gmres_restart 110 -ksp_rtol 1.0e-9   -ksp_max_it 1000
KSPOPT1 = $(KSPA) $(KSPB)

# ---------------------------------------------------------------------------------------------------------------------------------

# A_0 1st LOOP 
KS001 = -fieldsplit_0_ksp_type fgmres -fieldsplit_0_ksp_pc_side right -fieldsplit_0_ksp_rtol 1.0e-1 -fieldsplit_0_ksp_max_it 1000 
# FOR CHOLESKY & ICC  
# CHOLESKY        -fieldsplit_0_pc_type cholesky -fieldsplit_0_pc_factor_mat_solver_package mumps $(MUMPSOPT0)
# ICC             -fieldsplit_0_pc_type icc -fieldsplit_0_pc_factor_mat_solver_package petsc -fieldsplit_0_pc_factor_levels 100 -fieldsplit_0_pc_factor_in_place -fieldsplit_0_pc_factor_mat_ordering_type qmd
# JACOBI          -fieldsplit_0_pc_type jacobi -fieldsplit_0_pc_jacobi_type rowsum -fieldsplit_0_pc_jacobi_abs
# SOR             -fieldsplit_0_pc_type sor 	-fieldsplit_0_pc_sor_symmetric -fieldsplit_0_pc_sor_omega 1.0
# GAMG            -fieldsplit_0_pc_type gamg -pc_gamg_square_graph 30 -fieldsplit_0_pc_mg_type full -fieldsplit_0_pc_mg_smoothup 5 -fieldsplit_0_pc_gamg_threshold -1.0 
# HYPRE/PILUT     -fieldsplit_0_pc_type hypre -fieldsplit_0_pc_hypre_type pilut -fieldsplit_0_pc_hypre_pilut_tol 0.01 -fieldsplit_0_pc_factor_mat_ordering_type qmd 
# HYPRE/BOOMRAMG  -fieldsplit_0_pc_type hypre -fieldsplit_0_pc_hypre_type boomeramg -fieldsplit_0_pc_hypre_boomeramg_nodal_coarsen 5
# PFMG            -fieldsplit_0_pc_type pfmg -fieldsplit_0_pc_pfmg_num_pre_relax 2 -fieldsplit_0_pc_pfmg_tol 0.01 -fieldsplit_0_pc_pfmg_relax_type Jacobi
MUMPSOPT0 = -fieldsplit_0_mat_mumps_icntl_29 2 -fieldsplit_0_mat_mumps_icntl_7 0 -fieldsplit_0_mat_mumps_icntl_22 1 -fieldsplit_0_mat_mumps_icntl_24 1 -fieldsplit_0_mat_mumps_icntl_14 50 -fieldsplit_0_mat_mumps_cntl_3 1.0e-6
PC001 = -fieldsplit_0_pc_type gamg -pc_gamg_square_graph 30 -fieldsplit_0_pc_mg_type full -fieldsplit_0_pc_mg_smoothup 5 -fieldsplit_0_pc_gamg_threshold -1.0 

KSPOPT2 = $(KS001) $(PC001)

# General options :  #  -fieldsplit_0_ksp_converged_reason

# ---------------------------------------------------------------------------------------------------------------------------------

# A_shur 2nd LOOP 
KSShur = -fieldsplit_1_ksp_type preonly  -fieldsplit_1_ksp_pc_side right -fieldsplit_1_ksp_rtol 1.0e-1 -fieldsplit_1_ksp_max_it 1000 
PCShur = -fieldsplit_1_pc_type cholesky  -fieldsplit_1_pc_factor_mat_ordering_type nd 
#PCShur = -fieldsplit_1_pc_type hypre -fieldsplit_1_pc_hypre_type pilut -fieldsplit_1_pc_hypre_pilut_tol 0.01 -fieldsplit_1_pc_factor_mat_ordering_type qmd 
PCShur1 = -fieldsplit_1_inner_ksp_type preonly -fieldsplit_1_inner_pc_type jacobi
KSPOPT3 = $(KSShur) $(PCShur) $(PCShur1) 

# General options :  -fieldsplit_1_ksp_converged_reason

# ----------------------------------------------------------------------------------------------------------------------------
# Some other general options : -ksp_gmres_modifiedgramschmidt ksp_gmres_classicalgramschmidt -info 	
#  -info -ksp_type preonly -log_summary  -ksp_rtol 1.0e-7 	-ksp_max_it 200 -ksp_converged_reason -ksp_monitor_draw --draw_pause 2
#  -ksp_gmres_cgs_refinement_type refine_ifneeded	 -ksp_monitor_singular_value -ksp_compute_eigenvalues	-ksp_plot_eigenvalues 
# -pc_fieldsplit_schur_precondition selfp   -fieldsplit_1_pc_type lsc -fieldsplit_1_lsc_pc_type ml -fieldsplit_1_pc_jacobi_type diagonal 
# -pc_type hypre -pc_hypre_type euclid -pc_hypre_euclid_levels k

#
#  ---------------------------------------------------------------------------------------------------------------------------------
OPTGNR = -log_summary -ksp_converged_reason  -ksp_monitor

# Execution of the block resolution
DATASTORETEST1   = /petsc-3.7.6/Install/Test_pcfield/matrices/test_alternator
DATASTORETEST2   = /petsc-3.7.6/Install/Test_pcfield/matrices/test2
DATASTORETEST3   = /petsc-3.7.6/Install/Test_pcfield/matrices/test3
DATASTORETEST4   = /petsc-3.7.6/Install/Test_pcfield/matrices/test4
#  Needed file to construct the right hand side ----> [ 0 ; Z'f]
OPTForm1_TEST1 = -matZAZ $(DATASTORETEST1)/ZAZ.dat -matZBZ $(DATASTORETEST1)/ZBZ.dat -matZPrZ $(DATASTORETEST1)/ZPrZ.dat -vecrhs $(DATASTORETEST1)/rhs.dat
OPTForm1_TEST2 = -matZAZ $(DATASTORETEST2)/ZAZ.dat -matZBZ $(DATASTORETEST2)/ZBZ.dat -matZPrZ $(DATASTORETEST2)/ZPrZ.dat -vecrhs $(DATASTORETEST2)/rhs.dat
OPTForm1_TEST3 = -matZAZ $(DATASTORETEST3)/ZAZ.dat -matZBZ $(DATASTORETEST3)/ZBZ.dat -matZPrZ $(DATASTORETEST3)/ZPrZ.dat -vecrhs $(DATASTORETEST3)/rhs.dat
OPTForm1_TEST4 = -matZAZ $(DATASTORETEST4)/ZAZ.dat -matZBZ $(DATASTORETEST4)/ZBZ.dat -matZPrZ $(DATASTORETEST4)/ZPrZ.dat -vecrhs $(DATASTORETEST4)/rhs.dat

test0: 
		 -@${MPIEXEC} -n 1 ./main -user_pc $(KSPOPT1) $(KSPOPT2) $(KSPOPT3) $(OPTGNR)  $(OPTForm1_TEST1)  > alter_1proc_chol.tmp 2>&1;

test1: 
		 -@${MPIEXEC} -n 2 ./main -user_pc $(KSPOPT1) $(KSPOPT2) $(KSPOPT3) $(OPTGNR)  $(OPTForm1_TEST1)  > alter_2proc_chol.tmp 2>&1;


test2: 
		 -@${MPIEXEC} -n 4 ./main -user_pc $(KSPOPT1) $(KSPOPT2) $(KSPOPT3) $(OPTGNR)  $(OPTForm1_TEST1)  > alter_4proc_gamg.tmp 2>&1;


# -user_pc

include ${PETSC_DIR}/lib/petsc/conf/test
