import numpy

def my_contrasts(design_matrix):
    contrast_matrix = numpy.eye(design_matrix.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])
    
    # vigorization
    go_pmod = basic_contrasts['LeftGoH_pmod']+basic_contrasts['LeftGoL_pmod']+basic_contrasts['RightGoH_pmod']+basic_contrasts['RightGoL_pmod']

    rightgo_pmod = basic_contrasts['RightGoH_pmod']+basic_contrasts['RightGoL_pmod']
    leftgo_pmod = basic_contrasts['LeftGoH_pmod']+basic_contrasts['LeftGoL_pmod']
    
    # vigor lateralization
    leftright_pmod = (basic_contrasts['LeftGoH_pmod']+basic_contrasts['LeftGoL_pmod'])-(basic_contrasts['RightGoH_pmod']+basic_contrasts['RightGoL_pmod'])
    rightleft_pmod = (basic_contrasts['RightGoH_pmod']+basic_contrasts['RightGoL_pmod'])-(basic_contrasts['LeftGoH_pmod']+basic_contrasts['LeftGoL_pmod'])
    

    gohigh_pmod = basic_contrasts['LeftGoH_pmod'] + basic_contrasts['RightGoH_pmod']
    golow_pmod = basic_contrasts['LeftGoL_pmod'] + basic_contrasts['RightGoL_pmod']

    
    gohighleft_pmod = basic_contrasts['LeftGoH_pmod']
    gohighright_pmod = basic_contrasts['RightGoH_pmod']
    golowleft_pmod = basic_contrasts['LeftGoL_pmod']
    golowright_pmod = basic_contrasts['RightGoL_pmod']
    
    # reward context
    highlow_pmod = (basic_contrasts['RightGoH_pmod']+basic_contrasts['LeftGoH_pmod'])-(basic_contrasts['RightGoL_pmod']+basic_contrasts['LeftGoL_pmod'])
    lowhigh_pmod = (basic_contrasts['RightGoL_pmod']+basic_contrasts['LeftGoL_pmod'])-(basic_contrasts['RightGoH_pmod']+basic_contrasts['LeftGoH_pmod'])
    righthighlow_pmod = basic_contrasts['RightGoH_pmod']-basic_contrasts['RightGoL_pmod']
    lefthighlow_pmod = basic_contrasts['LeftGoH_pmod']-basic_contrasts['LeftGoL_pmod']
    


    contrasts = {
        'go_pmod':go_pmod,
        'rightgo_pmod':rightgo_pmod,
        'leftgo_pmod':leftgo_pmod,

        'leftright_pmod':leftright_pmod,
        'rightleft_pmod':rightleft_pmod,
        
        'gohigh_pmod':gohigh_pmod,
        'golow_pmod':golow_pmod,
        
        'gohighleft_pmod':gohighleft_pmod,
        'gohighright_pmod':gohighright_pmod,
        'golowleft_pmod':golowleft_pmod,
        'golowright_pmod':golowright_pmod,
        
        'highlow_pmod':highlow_pmod,
        'lowhigh_pmod':lowhigh_pmod,
        
        'righthighlow_pmod':righthighlow_pmod,
        'lefthighlow_pmod':lefthighlow_pmod,
        
    }
    
    # Calculate weights on the contrasts
    for key, array in contrasts.items():
        positive_sum = numpy.sum(array[array > 0])
        negative_sum = numpy.sum(array[array < 0])
        
        if positive_sum != 0:
            array[array > 0] /= positive_sum
        if negative_sum != 0:
            array[array < 0] /= abs(negative_sum)
        
        contrasts[key] = array

    return basic_contrasts, contrasts