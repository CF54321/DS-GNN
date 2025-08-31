import os
import shutil

file_A_path = 'E:/Data/asd_new/pearson_mat_Reho_fALLL_ALLL_DC_VHMC'
file_A_outpath = 'E:/Data/asd_new/Sorted_pearson_mat_5'
file_B_path = 'E:/Data/asd_new/f_MRIdata/Pearson_matrix_fMRI_ROI_AAL'
file_B_outpath = 'E:/Data/asd_new/Pearson_matrix_fMRI_ROI_AAL'
file_A_list = os.listdir(file_A_path)
file_B_list = os.listdir(file_B_path)
file_A_num = [i[-9:-4] for i in file_A_list]
file_B_num = [i[-9:-4] for i in file_B_list]
for i in file_A_num:
    if i in file_B_num:
        shutil.copy(file_B_path + '/' + '00{}.csv'.format(int(i)), file_B_outpath + '/' + i + '.csv')
        # shutil.copy(file_B_path + '/' + 'JS_KSDENSITY_256_Node_feature_Signal_sub_{}.mat'.format(int(i))
        #             , file_B_outpath + '/' + i + '.mat')
