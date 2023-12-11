# Working directory path to PC m-types  
path = 'data/'
cell_type = 'Reconstructed_PCs/'

pop_l2_ipc_pwd = path + cell_type + 'L2/L2_IPC'
pop_l2_tpc_a_pwd = path + cell_type + 'L2/L2_TPC:A'
pop_l2_tpc_b_pwd = path + cell_type + 'L2/L2_TPC:B'
PC_pwd_l2_list = [pop_l2_ipc_pwd, pop_l2_tpc_a_pwd, pop_l2_tpc_b_pwd]

pop_l3_tpc_a_pwd = path + cell_type + 'L3/L3_TPC:A'
pop_l3_tpc_c_pwd = path + cell_type + 'L3/L3_TPC:C'
PC_pwd_l3_list = [pop_l3_tpc_a_pwd, pop_l3_tpc_c_pwd]

pop_l4_tpc_pwd = path + cell_type + 'L4/L4_TPC'
pop_l4_upc_pwd = path + cell_type + 'L4/L4_UPC'
PC_pwd_l4_list = [pop_l4_tpc_pwd, pop_l4_upc_pwd]

pop_l5_tpc_a_pwd = path + cell_type + 'L5/L5_TPC:A'
pop_l5_tpc_b_pwd = path + cell_type + 'L5/L5_TPC:B'
pop_l5_tpc_c_pwd = path + cell_type + 'L5/L5_TPC:C'
pop_l5_upc_pwd = path + cell_type + 'L5/L5_UPC'
PC_pwd_l5_list = [pop_l5_tpc_a_pwd, pop_l5_tpc_b_pwd, pop_l5_tpc_c_pwd, pop_l5_upc_pwd]

pop_l6_bpc_pwd = path + cell_type + 'L6/L6_BPC'
pop_l6_hpc_pwd = path + cell_type + 'L6/L6_HPC'
pop_l6_ipc_pwd = path + cell_type + 'L6/L6_IPC'
pop_l6_tpc_a_pwd = path + cell_type + 'L6/L6_TPC:A'
pop_l6_tpc_c_pwd = path + cell_type + 'L6/L6_TPC:C'
pop_l6_upc_pwd = path + cell_type + 'L6/L6_UPC'
PC_pwd_l6_list = [pop_l6_bpc_pwd, pop_l6_hpc_pwd, pop_l6_ipc_pwd, pop_l6_tpc_a_pwd, pop_l6_tpc_c_pwd, pop_l6_upc_pwd]

PC_pwd_list = {'L2':PC_pwd_l2_list,
            'L3':PC_pwd_l3_list,
             'L4':PC_pwd_l4_list,
              'L5':PC_pwd_l5_list,
               'L6':PC_pwd_l6_list}