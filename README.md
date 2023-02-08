# Nuclear-spin-Hyperpolarization
This project contains the code and notes that I have used to study spin-fluctuator model. 
      ![img.png](img.png)



My notes:(..%2F..%2FDesktop%2FHyperpolarization%20of%20Nuclear%20spin_original.pptx)
1. Concentration of NV: 1ppm
2. Paramagnetic impurities P1: >20 ppm
3. Roles of P1 centers (in terms of charge ionization): they provides -ve electrons to NV center to make NV-charge state.

# Compute spin dephasing time
1. Make supercell of NV- and do structural relaxation. Then perform scf calculation to get the relaxed structure.
2. Compute zero field splitting tensor and hyperfine tensor for defect ground state, defect excited state and ionized defect.
3. Calculate transition dipole matrix element between NV- excited state to the conduction band. In order to compute transition dipole matrix element, use VASPKIT.

