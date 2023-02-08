import numpy as np
#import qutip as qp
import matplotlib as plt

############# Parameters ################################################################
### Ground state triplet ###########
D_gs=2.87    # in GHZ Zero Field Splitting of ground state
gm_e_parl=2.8e-3    # in GHZ/G is the parallel component of gyromagnetic ratio for electron
gm_e_perp=2.8e-3    # in GHZ/G is the perpendicular component of gyromagnetic ratio for electron
B_z= 30e-3        # in T parallel component of magnetic field
B_x=30e-3      # in T X(perpendicular) component of magnetic field
B_y=30e-3     # in T(perpendicular) component of magnetic field

### Excited state triplet ###########
D_es=1.42  # in GHZ Zero Field Splitting of ground state
# gm_e_parl, gm_e_perp, B_z, B_x and B_y are same

###########################################################################################
#### Hamiltonian ###################################
H=np.zeros((7,7),dtype=complex)
H[0,0]=1/3*D_gs+gm_e_parl*B_z
H[0,1]=1/np.sqrt(2)*gm_e_perp*(B_x-1.0J*B_y)
H[1,0]=1/np.sqrt(2)*gm_e_perp*(B_x+1.0j*B_y)
H[1,1]=-2/3*D_gs
H[1,2]=1/np.sqrt(2)*gm_e_perp*(B_x-1.0j*B_y)
H[2,1]=1/np.sqrt(2)*gm_e_perp*(B_x+1.0j*B_y)
H[2,2]=1/3*D_gs-gm_e_parl*B_z
H[3,3]=1/3*D_es+gm_e_parl*B_z
H[3,4]=1/np.sqrt(2)*gm_e_perp*(B_x-1.0j*B_y)
H[4,3]=1/np.sqrt(2)*gm_e_perp*(B_x+1.0j*B_y)
H[4,4]=-2/3*D_es
H[4,5]=1/np.sqrt(2)*gm_e_perp*(B_x-1.0j*B_y)
H[5,4]=1/np.sqrt(2)*gm_e_perp*(B_x+1.0j*B_y)
H[5,5]=1/3*D_es-gm_e_parl*B_z

###############################################################################################
######  Decay rates  ##################################
R=0.961 #in MHZ Pumping rate
k_eg=83.33 ## in MHZ decay rare from ES to GS (same for 3 different spin conserving transitions)
k_ge=R*k_eg  ##in MHZ excitation rate from GS to ES
k_nr=83.33  ##in MHZ decay rate from excited state to singlet state
k_m0=7.0   ## in MHZ decay rate from singlet to ground state

k_phi=10.0e5  ## MHZ dephasing rates of excites state triplets

#############################################################################################
################### Lindblad operators: ######################################################
##############################################################################################
##############      Relaxation operators #####################################################

L_14=np.zeros((7,7),dtype=complex)
L_14[3,0]=k_ge
L_41=np.zeros((7,7),dtype=complex)
L_41[0,3]=k_eg

L_25=np.zeros((7,7),dtype=complex)
L_25[4,1]=k_ge
L_52=np.zeros((7,7),dtype=complex)
L_52[1,4]=k_eg

L_36=np.zeros((7,7),dtype=complex)
L_36[5,2]=k_ge
L_63=np.zeros((7,7),dtype=complex)
L_63[2,5]=k_eg

L_47=np.zeros((7,7),dtype=complex)
L_47[6,3]=k_nr

L_57=np.zeros((7,7),dtype=complex)
L_57[6,4]=k_nr

L_67=np.zeros((7,7),dtype=complex)
L_67[6,5]=k_nr

L_71=np.zeros((7,7),dtype=complex)
L_71[0,6]=k_m0

L_72=np.zeros((7,7),dtype=complex)
L_72[1,6]=k_m0

L_73=np.zeros((7,7),dtype=complex)
L_73[2,6]=k_m0

######################################################################################################
################# Dephasing operators ##############################################################
L_44=np.zeros((7,7),dtype=complex)
L_44[3,3]=k_phi

L_55=np.zeros((7,7),dtype=complex)
L_55[4,4]=k_phi

L_66=np.zeros((7,7),dtype=complex)
L_66[5,5]=k_phi

#####################################################################################################
#####################################################################################################
############ Initial density matrix  ############################
################################################################
rho_0=np.zeros((7,7),dtype=complex)
rho_0[0,0]=1/3
rho_0[1,1]=1/3
rho_0[2,2]=1/3
rho_vec=rho_0.reshape(49,1)

#####################################################################################################
I_mat=np.eye(7, dtype=complex)
##################################################################################
## Livullian operator ##########################
L_Luvillian= 1.0j*np.kron(H.conj(),I_mat)-1.0j*np.kron(I_mat,H)

D_L_14=np.kron(L_14.conj(),L_14)-0.5*np.kron(I_mat,np.matmul(L_14.conj().T,L_14))-0.5*np.kron(np.matmul(L_14.conj().T,L_14),I_mat)

D_L_41=np.kron(L_41.conj(),L_41)-0.5*np.kron(I_mat,np.matmul(L_41.conj().T,L_41))-0.5*np.kron(np.matmul(L_41.conj().T,L_41),I_mat)

D_L_25=np.kron(L_25.conj(),L_25)-0.5*np.kron(I_mat,np.matmul(L_25.conj().T,L_25))-0.5*np.kron(np.matmul(L_25.conj().T,L_25),I_mat)

D_L_52=np.kron(L_52.conj(),L_52)-0.5*np.kron(I_mat,np.matmul(L_52.conj().T,L_52))-0.5*np.kron(np.matmul(L_52.conj().T,L_52),I_mat)

D_L_63=np.kron(L_63.conj(),L_63)-0.5*np.kron(I_mat,np.matmul(L_63.conj().T,L_63))-0.5*np.kron(np.matmul(L_63.conj().T,L_63),I_mat)

D_L_36=np.kron(L_36.conj(),L_36)-0.5*np.kron(I_mat,np.matmul(L_36.conj().T,L_36))-0.5*np.kron(np.matmul(L_36.conj().T,L_36),I_mat)

D_L_47=np.kron(L_47.conj(),L_47)-0.5*np.kron(I_mat,np.matmul(L_47.conj().T,L_47))-0.5*np.kron(np.matmul(L_47.conj().T,L_47),I_mat)

D_L_57=np.kron(L_57.conj(),L_57)-0.5*np.kron(I_mat,np.matmul(L_57.conj().T,L_57))-0.5*np.kron(np.matmul(L_57.conj().T,L_57),I_mat)

D_L_67=np.kron(L_67.conj(),L_67)-0.5*np.kron(I_mat,np.matmul(L_67.conj().T,L_67))-0.5*np.kron(np.matmul(L_67.conj().T,L_67),I_mat)

D_L_71=np.kron(L_71.conj(),L_71)-0.5*np.kron(I_mat,np.matmul(L_71.conj().T,L_71))-0.5*np.kron(np.matmul(L_71.conj().T,L_71),I_mat)

D_L_72=np.kron(L_72.conj(),L_72)-0.5*np.kron(I_mat,np.matmul(L_72.conj().T,L_72))-0.5*np.kron(np.matmul(L_72.conj().T,L_72),I_mat)

D_L_73=np.kron(L_73.conj(),L_73)-0.5*np.kron(I_mat,np.matmul(L_73.conj().T,L_73))-0.5*np.kron(np.matmul(L_73.conj().T,L_73),I_mat)

##############################
D_L_44=np.kron(L_44.conj(),L_44)-0.5*np.kron(I_mat,np.matmul(L_44.conj().T,L_44))-0.5*np.kron(np.matmul(L_44.conj().T,L_44),I_mat)

D_L_55=np.kron(L_55.conj(),L_55)-0.5*np.kron(I_mat,np.matmul(L_55.conj().T,L_55))-0.5*np.kron(np.matmul(L_55.conj().T,L_55),I_mat)

D_L_66=np.kron(L_66.conj(),L_66)-0.5*np.kron(I_mat,np.matmul(L_66.conj().T,L_66))-0.5*np.kron(np.matmul(L_66.conj().T,L_66),I_mat)

##################################################################################################################################
##################################################################################################################################

L_SuOpt=L_Luvillian + D_L_14 + D_L_41 + D_L_25 + D_L_52 + D_L_36 + D_L_63 + D_L_47+D_L_57+D_L_67+D_L_71+D_L_72+D_L_73+D_L_44+D_L_55+D_L_66
