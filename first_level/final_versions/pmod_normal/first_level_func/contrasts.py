import numpy

def my_contrasts(design_matrix):
    contrast_matrix = numpy.eye(design_matrix.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])
    
    # vigorization
    # Where does brain activity scale with trial-by-trial activity across all conditions, averaged across all participants?
    go_pmod = basic_contrasts['LeftGoH_pmod']+basic_contrasts['LeftGoL_pmod']+basic_contrasts['RightGoH_pmod']+basic_contrasts['RightGoL_pmod']
    rightgo_pmod = basic_contrasts['RightGoH_pmod']+basic_contrasts['RightGoL_pmod']
    leftgo_pmod = basic_contrasts['LeftGoH_pmod']+basic_contrasts['LeftGoL_pmod']
    
    go = basic_contrasts['LeftGoH']+basic_contrasts['LeftGoL']+basic_contrasts['RightGoH']+basic_contrasts['RightGoL']
    leftgo = basic_contrasts['LeftGoH']+basic_contrasts['LeftGoL']
    rightgo = basic_contrasts['RightGoH']+basic_contrasts['RightGoL']
    
    # vigor lateralization
    leftright_pmod = (basic_contrasts['LeftGoH_pmod']+basic_contrasts['LeftGoL_pmod'])-(basic_contrasts['RightGoH_pmod']+basic_contrasts['RightGoL_pmod'])
    rightleft_pmod = (basic_contrasts['RightGoH_pmod']+basic_contrasts['RightGoL_pmod'])-(basic_contrasts['LeftGoH_pmod']+basic_contrasts['LeftGoL_pmod'])
    
    leftright = (basic_contrasts['LeftGoH']+basic_contrasts['LeftGoL'])-(basic_contrasts['RightGoH']+basic_contrasts['RightGoL'])
    rightleft = (basic_contrasts['RightGoH']+basic_contrasts['RightGoL'])-(basic_contrasts['LeftGoH']+basic_contrasts['LeftGoL'])
    
    leftrighthigh_pmod = basic_contrasts['LeftGoH_pmod']-basic_contrasts['RightGoH_pmod']
    rightlefthigh_pmod = -basic_contrasts['LeftGoH_pmod']+basic_contrasts['RightGoH_pmod']
    leftrightlow_pmod = basic_contrasts['LeftGoL_pmod']-basic_contrasts['RightGoL_pmod']
    rightleftlow_pmod = -basic_contrasts['LeftGoL_pmod']+basic_contrasts['RightGoL_pmod']
    
    leftrighthigh = basic_contrasts['LeftGoH']-basic_contrasts['RightGoH']
    rightlefthigh = -basic_contrasts['LeftGoH']+basic_contrasts['RightGoH']
    leftrightlow = basic_contrasts['LeftGoL']-basic_contrasts['RightGoL']
    rightleftlow = -basic_contrasts['LeftGoL']+basic_contrasts['RightGoL']
    

    gohigh_pmod = basic_contrasts['LeftGoH_pmod'] + basic_contrasts['RightGoH_pmod']
    golow_pmod = basic_contrasts['LeftGoL_pmod'] + basic_contrasts['RightGoL_pmod']
    gohigh = basic_contrasts['LeftGoH'] + basic_contrasts['RightGoH']
    golow = basic_contrasts['LeftGoL'] + basic_contrasts['RightGoL']

    
    gohighleft_pmod = basic_contrasts['LeftGoH_pmod']
    gohighright_pmod = basic_contrasts['RightGoH_pmod']
    golowleft_pmod = basic_contrasts['LeftGoL_pmod']
    golowright_pmod = basic_contrasts['RightGoL_pmod']
    
    gohighleft = basic_contrasts['LeftGoH']
    gohighright = basic_contrasts['RightGoH']
    golowleft = basic_contrasts['LeftGoL']
    golowright = basic_contrasts['RightGoL']
    
    
    # reward context
    highlow_pmod = (basic_contrasts['RightGoH_pmod']+basic_contrasts['LeftGoH_pmod'])-(basic_contrasts['RightGoL_pmod']+basic_contrasts['LeftGoL_pmod'])
    lowhigh_pmod = (basic_contrasts['RightGoL_pmod']+basic_contrasts['LeftGoL_pmod'])-(basic_contrasts['RightGoH_pmod']+basic_contrasts['LeftGoH_pmod'])
    highlow = (basic_contrasts['RightGoH']+basic_contrasts['LeftGoH'])-(basic_contrasts['RightGoL']+basic_contrasts['LeftGoL'])
    lowhigh = (basic_contrasts['RightGoL']+basic_contrasts['LeftGoL'])-(basic_contrasts['RightGoH']+basic_contrasts['LeftGoH'])
    
    righthighlow_pmod = basic_contrasts['RightGoH_pmod']-basic_contrasts['RightGoL_pmod']
    lefthighlow_pmod = basic_contrasts['LeftGoH_pmod']-basic_contrasts['LeftGoL_pmod']
    rightlowhigh_pmod = -basic_contrasts['RightGoH_pmod']+basic_contrasts['RightGoL_pmod']
    leftlowhigh_pmod = -basic_contrasts['LeftGoH_pmod']+basic_contrasts['LeftGoL_pmod']
    
    righthighlow = basic_contrasts['RightGoH']-basic_contrasts['RightGoL']
    lefthighlow = basic_contrasts['LeftGoH']-basic_contrasts['LeftGoL']
    rightlowhigh = -basic_contrasts['RightGoH']+basic_contrasts['RightGoL']
    leftlowhigh = -basic_contrasts['LeftGoH']+basic_contrasts['LeftGoL']
    


    contrasts = {
        'go_pmod':go_pmod,
        'rightgo_pmod':rightgo_pmod,
        'leftgo_pmod':leftgo_pmod,
        
        'go':go,
        'rightgo':rightgo,
        'leftgo':leftgo,

        'leftright_pmod':leftright_pmod,
        'rightleft_pmod':rightleft_pmod,
        
        'leftright':leftright,
        'rightleft':rightleft,
        
        'gohigh_pmod':gohigh_pmod,
        'golow_pmod':golow_pmod,
        
        'gohigh':gohigh,
        'golow':golow,
        
        'gohighleft_pmod':gohighleft_pmod,
        'gohighright_pmod':gohighright_pmod,
        'golowleft_pmod':golowleft_pmod,
        'golowright_pmod':golowright_pmod,
        
        'gohighleft':gohighleft,
        'gohighright':gohighright,
        'golowleft':golowleft,
        'golowright':golowright,
        
        'highlow_pmod':highlow_pmod,
        'lowhigh_pmod':lowhigh_pmod,
        
        'highlow':highlow,
        'lowhigh':lowhigh,
        
        'righthighlow_pmod':righthighlow_pmod,
        'lefthighlow_pmod':lefthighlow_pmod,
        'rightlowhigh_pmod':rightlowhigh_pmod,
        'leftlowhigh_pmod':leftlowhigh_pmod,
        
        'righthighlow':righthighlow,
        'lefthighlow':lefthighlow,
        'rightlowhigh':rightlowhigh,
        'leftlowhigh':leftlowhigh,
        
        'leftrighthigh_pmod':leftrighthigh_pmod,
        'rightlefthigh_pmod':rightlefthigh_pmod,
        'leftrightlow_pmod':leftrightlow_pmod,
        'rightleftlow_pmod':rightleftlow_pmod,
        
        'leftrighthigh':leftrighthigh,
        'rightlefthigh':rightlefthigh,
        'leftrightlow':leftrightlow,
        'rightleftlow':rightleftlow
        
    }
    
    # Calculate weights on the contrasts
    
    # for key, array in contrasts.items():
    #     positive_sum = numpy.sum(array[array > 0])
    #     negative_sum = numpy.sum(array[array < 0])
        
    #     if positive_sum != 0:
    #         array[array > 0] /= positive_sum
    #     if negative_sum != 0:
    #         array[array < 0] /= abs(negative_sum)
        
    #     contrasts[key] = array

    return basic_contrasts, contrasts