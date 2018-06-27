
PETSC_DIR        = /home/J08475/petsc-3.7.6/Install
LOCDIR           = /home/J08475/petsc-3.7.6/Install
EXAMPLESC        = test.c 
DATAFILESPATH    = /home/J08475/petsc-3.7.6/Install/Test_shell
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
DATAFILESPATH1    = ~/petsc-3.7.6/Install/Test_pcfield/matrices/test5
# Where to store the saddle point matrix blocks 
DATASTORE   = ~/petsc-3.7.6/Install/Test_pcfield/matrices/test5
# Needed files to construct the saddle point matrix blocks
DATAFILESstore =  -matM $(DATAFILESPATH1)/matM -matK $(DATAFILESPATH1)/matK -matprod $(DATAFILESPATH1)/matprod -matZ $(DATAFILESPATH1)/matZ -matobs $(DATAFILESPATH1)/matobs  -matchKr $(DATAFILESPATH1)/matchKr

# Choose the saddle point problem in form 1 or form 2
OPTForm1 = -form1 -matZAZ $(DATASTORE)/ZAZ.dat -matZBZ $(DATASTORE)/ZBZ.dat -matZPrZ $(DATASTORE)/ZPrZ.dat
OPTForm2 = -form2 -matH $(DATASTORE)/H.dat  -matF $(DATASTORE)/F.dat

# Execution of the storing command 
run_store: 
		-@${MPIEXEC}  -n 1 ./store  $(DATAFILESstore) $(OPTForm1) > result1.tmp 2>&1;


#---------------------------------------------------------------------------------------------
#           Test block resolution in Form 1 (ZAZ, ZBZ and ZPrZ ) or Form 2  (H and F)        #
#---------------------------------------------------------------------------------------------

# Compilation 
csrc = $(wildcard src/*.c) 
obj = $(csrc: .c=.o)
main: $(obj) 
	-${CLINKER} -o  $@ $^ ${PETSC_KSP_LIB}

#  Needed file to construct the right hand side ----> [ 0 ; Z'f]
DATAFILES11 =  -matZ $(DATAFILESPATH1)/matZ -rhs $(DATAFILESPATH1)/Rhs

#  Solver runtime options  ! note that if you chose an other type of solver, this will override the block resolution
# Choose a preconditioner shur or lagrangian 
OUTERSOL = -lagrangian  -ksp_gmres_restart 10000 -ksp_rtol 1.0e-6 -ksp_max_it 1000
OPTGNR = -log_summary  -ksp_converged_reason  -ksp_monitor_true_residual   -ksp_norm_type unpreconditioned  -ksp_monitor_lg_residualnorm
# Some options : -ksp_monitor_true_residual  -ksp_norm_type unpreconditioned -info 	-ksp_monitor_lg_residualnorm
#       -info -ksp_type preonly -log_summary  -ksp_rtol 1.0e-7 	-ksp_max_it 200 -ksp_converged_reason


# Execution of the block resolution
case: 
		-@${MPIEXEC}  -n 1 ./main  $(OUTERSOL) $(OPTGNR) $(DATAFILES11) $(OPTForm2)  > result1.tmp 2>&1;








include ${PETSC_DIR}/lib/petsc/conf/test
