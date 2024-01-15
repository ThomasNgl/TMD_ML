# Working directory path to IN m-types  
path = '/Users/thomasnegrello/Documents/Lida/TMD_ML/data/'
cell_type = 'Reconstructed_INs/'

pop_l1_dac_pwd = path + cell_type + 'L1/L1_DAC'
pop_l1_hac_pwd = path + cell_type + 'L1/L1_HAC'
pop_l1_lac_pwd = path + cell_type + 'L1/L1_LAC'
pop_l1_ngc_da_pwd = path + cell_type + 'L1/L1_NGC-DA'
pop_l1_ngc_sa_pwd = path + cell_type + 'L1/L1_NGC-SA'
pop_l1_sac_pwd = path + cell_type + 'L1/L1_SAC'
IN_pwd_l1_list = [pop_l1_dac_pwd, pop_l1_hac_pwd, pop_l1_lac_pwd, pop_l1_ngc_da_pwd, pop_l1_ngc_sa_pwd, pop_l1_sac_pwd]

#pop_l23_bp_pwd = path + cell_type + 'L23/L23_BP'
pop_l23_btc_pwd = path + cell_type + 'L23/L23_BTC'
pop_l23_chc_pwd = path + cell_type + 'L23/L23_CHC'
pop_l23_dbc_pwd = path + cell_type + 'L23/L23_DBC'
pop_l23_lbc_pwd = path + cell_type + 'L23/L23_LBC'
pop_l23_mc_pwd = path + cell_type + 'L23/L23_MC'
pop_l23_nbc_pwd = path + cell_type + 'L23/L23_NBC'
pop_l23_ngc_pwd = path + cell_type + 'L23/L23_NGC'
pop_l23_sbc_pwd = path + cell_type + 'L23/L23_SBC'
IN_pwd_l23_list = [pop_l23_btc_pwd, pop_l23_chc_pwd, pop_l23_dbc_pwd, pop_l23_lbc_pwd, pop_l23_mc_pwd,
                   pop_l23_nbc_pwd, pop_l23_ngc_pwd, pop_l23_sbc_pwd]

#pop_l4_bp_pwd = path + cell_type + 'L4/L4_BP'
pop_l4_btc_pwd = path + cell_type + 'L4/L4_BTC'
#pop_l4_chc_pwd = path + cell_type + 'L4/L4_CHC'
pop_l4_dbc_pwd = path + cell_type + 'L4/L4_DBC'
pop_l4_lbc_pwd = path + cell_type + 'L4/L4_LBC'
pop_l4_mc_pwd = path + cell_type + 'L4/L4_MC'
pop_l4_nbc_pwd = path + cell_type + 'L4/L4_NBC'
#pop_l4_ngc_pwd = path + cell_type + 'L4/L4_NGC'
pop_l4_sbc_pwd = path + cell_type + 'L4/L4_SBC'
IN_pwd_l4_list = [pop_l4_btc_pwd, pop_l4_dbc_pwd, pop_l4_lbc_pwd, pop_l4_mc_pwd, pop_l4_nbc_pwd, pop_l4_sbc_pwd]

pop_l5_bp_pwd = path + cell_type + 'L5/L5_BP'
pop_l5_btc_pwd = path + cell_type + 'L5/L5_BTC'
pop_l5_chc_pwd = path + cell_type + 'L5/L5_CHC'
pop_l5_dbc_pwd = path + cell_type + 'L5/L5_DBC'
pop_l5_lbc_pwd = path + cell_type + 'L5/L5_LBC'
pop_l5_mc_pwd = path + cell_type + 'L5/L5_MC'
pop_l5_nbc_pwd = path + cell_type + 'L5/L5_NBC'
pop_l5_sbc_pwd = path + cell_type + 'L5/L5_SBC'
IN_pwd_l5_list = [pop_l5_bp_pwd, pop_l5_btc_pwd, pop_l5_chc_pwd, pop_l5_dbc_pwd, pop_l5_lbc_pwd, pop_l5_mc_pwd, pop_l5_nbc_pwd, pop_l5_sbc_pwd]

#pop_l6_btc_pwd = path + cell_type + 'L6/L6_BTC'
#pop_l6_chc_pwd = path + cell_type + 'L6/L6_CHC'
#pop_l6_dbc_pwd = path + cell_type + 'L6/L6_DBC'
pop_l6_lbc_pwd = path + cell_type + 'L6/L6_LBC'
pop_l6_mc_pwd = path + cell_type + 'L6/L6_MC'
pop_l6_nbc_pwd = path + cell_type + 'L6/L6_NBC'
#pop_l6_ngc_pwd = path + cell_type + 'L6/L6_NGC'
pop_l6_sbc_pwd = path + cell_type + 'L6/L6_SBC'
IN_pwd_l6_list = [
pop_l6_lbc_pwd,
pop_l6_mc_pwd ,
pop_l6_nbc_pwd,
pop_l6_sbc_pwd]

IN_pwd_list = {'L1':IN_pwd_l1_list,
            'L23':IN_pwd_l23_list,
             'L4':IN_pwd_l4_list,
              'L5':IN_pwd_l5_list,
               'L6':IN_pwd_l6_list}