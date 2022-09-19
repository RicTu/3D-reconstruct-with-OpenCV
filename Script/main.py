from functions import *

if __name__ == '__main__':
    # Step1: Get camera matrix
    pair = "R_L" # R_B, L_B, R_L
    #start_camera_calibrate(pair)
    #Step2: Cooresponding points
    # 2D tracling result
    info = ["05571583(second)_R_B_deflicker_in_206_160_preds",
            "05571583(second)_R_L_deflicker_in_179_81_preds",
            "05571583(second)_R_B_deflicker_th_260_177_preds",
            "05571583(second)_R_L_deflicker_th_186_163_preds"]
    #Step3: Calculate 3D information
    calculate_3d_datas_and_feature(info, pair)
    
