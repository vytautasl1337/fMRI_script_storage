import numpy

def my_contrasts(design_matrix):
    contrast_matrix = numpy.eye(design_matrix.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])
    
    go = basic_contrasts['Left']+basic_contrasts['Right']
    left = basic_contrasts['Left']
    right = basic_contrasts['Right']
    
    leftright = basic_contrasts['Left']-basic_contrasts['Right']
    rightleft = basic_contrasts['Right']-basic_contrasts['Left']
    
    go_slope = basic_contrasts['Left_slope_pmod']+basic_contrasts['Right_slope_pmod']
    go_slope_decrease = -basic_contrasts['Left_slope_pmod']-basic_contrasts['Right_slope_pmod']
    
    left_slope = basic_contrasts['Left_slope_pmod']
    right_slope = basic_contrasts['Right_slope_pmod']
    
    left_slope_decrease = -basic_contrasts['Left_slope_pmod']
    right_slope_decrease = -basic_contrasts['Right_slope_pmod']
    
    highlow = basic_contrasts['Reward_prospect_pmod']
    lowhigh = -basic_contrasts['Reward_prospect_pmod']
    
    
    


    contrasts = {
        'go':go,
        'left':left,
        'right':right,
        'rightleft':rightleft,
        'leftright':leftright,
        'go_slope':go_slope,
        'go_slope_decrease':go_slope_decrease,
        'left_slope':left_slope,
        'right_slope':right_slope,
        'left_slope_decrease':left_slope_decrease,
        'right_slope_decrease':right_slope_decrease,
        'highlow':highlow,
        'lowhigh':lowhigh
    }
    


    return basic_contrasts, contrasts