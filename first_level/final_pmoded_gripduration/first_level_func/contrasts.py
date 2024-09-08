import numpy

def my_contrasts(design_matrix):
    contrast_matrix = numpy.eye(design_matrix.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])
    
    # vigorization
    go_pmod = basic_contrasts['LeftGoH_pmod']+basic_contrasts['LeftGoL_pmod']+basic_contrasts['RightGoH_pmod']+basic_contrasts['RightGoL_pmod']
    go = basic_contrasts['LeftGoH']+basic_contrasts['LeftGoL']+basic_contrasts['RightGoH']+basic_contrasts['RightGoL']
    gonogo = (basic_contrasts['LeftGoL'] + basic_contrasts['RightGoL'] +
              basic_contrasts['LeftGoH'] + basic_contrasts['RightGoH']) - basic_contrasts['NoGoH'] - basic_contrasts['NoGoL']
    
    # vigor lateralization
    leftright_pmod = basic_contrasts['LeftGoH_pmod']+basic_contrasts['LeftGoL_pmod']-basic_contrasts['RightGoH_pmod']-basic_contrasts['RightGoL_pmod']
    leftright = basic_contrasts['LeftGoH']+basic_contrasts['LeftGoL']-basic_contrasts['RightGoH']-basic_contrasts['RightGoL']
    rightleft_pmod = basic_contrasts['RightGoH_pmod']+basic_contrasts['RightGoL_pmod']-basic_contrasts['LeftGoH_pmod']-basic_contrasts['LeftGoL_pmod']
    rightleft = basic_contrasts['RightGoH']+basic_contrasts['RightGoL']-basic_contrasts['LeftGoH']-basic_contrasts['LeftGoL']
    
    # inhibition
    nogo = basic_contrasts['NoGoH']+basic_contrasts['NoGoL']
    nogogo = (basic_contrasts['NoGoH'] + basic_contrasts['NoGoL'])-(basic_contrasts['LeftGoL'] + basic_contrasts['RightGoL'] +
              basic_contrasts['LeftGoH'] + basic_contrasts['RightGoH'])
    
    
    gohigh_pmod = basic_contrasts['LeftGoH_pmod'] + basic_contrasts['RightGoH_pmod']
    golow_pmod = basic_contrasts['LeftGoL_pmod'] + basic_contrasts['RightGoL_pmod']
    gohigh = basic_contrasts['RightGoH']+basic_contrasts['LeftGoH']
    golow = basic_contrasts['RightGoL']+basic_contrasts['LeftGoL']
    
    gohighleft_pmod = basic_contrasts['LeftGoH_pmod']
    gohighleft = basic_contrasts['LeftGoH']
    gohighright_pmod = basic_contrasts['RightGoH_pmod']
    gohighright = basic_contrasts['RightGoH']
    golowleft_pmod = basic_contrasts['LeftGoL_pmod']
    golowleft = basic_contrasts['LeftGoL']
    golowright_pmod = basic_contrasts['RightGoL_pmod']
    golowright = basic_contrasts['RightGoL']
    
    # reward context
    highlow_pmod = basic_contrasts['RightGoH_pmod']+basic_contrasts['LeftGoH_pmod']-basic_contrasts['RightGoL_pmod']-basic_contrasts['LeftGoL_pmod']
    lowhigh_pmod = -basic_contrasts['RightGoH_pmod']-basic_contrasts['LeftGoH_pmod']+basic_contrasts['RightGoL_pmod']+basic_contrasts['LeftGoL_pmod']
    righthighlow_pmod = basic_contrasts['RightGoH_pmod']-basic_contrasts['RightGoL_pmod']
    lefthighlow_pmod = basic_contrasts['LeftGoH_pmod']-basic_contrasts['LeftGoL_pmod']
    
    highlow = basic_contrasts['RightGoH']+basic_contrasts['LeftGoH']-basic_contrasts['RightGoL']-basic_contrasts['LeftGoL']
    lowhigh = -basic_contrasts['RightGoH']-basic_contrasts['LeftGoH']+basic_contrasts['RightGoL']+basic_contrasts['LeftGoL']
    righthighlow = basic_contrasts['RightGoH']-basic_contrasts['RightGoL']
    lefthighlow = basic_contrasts['LeftGoH']-basic_contrasts['LeftGoL']
    

    
    # reward
    win = basic_contrasts['Win']
    loss = basic_contrasts['Loss']
    winloss = basic_contrasts['Win']-basic_contrasts['Loss']
    losswin = basic_contrasts['Loss']-basic_contrasts['Win']
    
    # emotion
    sadneutral = basic_contrasts['Sad'] - basic_contrasts['Neutral']
    neutralsad = basic_contrasts['Neutral'] - basic_contrasts['Sad']
    sadhappy = basic_contrasts['Sad'] - basic_contrasts['Happy']
    happysad = basic_contrasts['Happy']- basic_contrasts['Sad']
    happyneutral = basic_contrasts['Happy'] - basic_contrasts['Neutral']
    neutralhappy = basic_contrasts['Neutral']- basic_contrasts['Happy']


    contrasts = {
        'go':go,
        'go_pmod':go_pmod,
        'gonogo': gonogo,
        'nogogo':nogogo,
        'nogo':nogo,
        
        'leftright_pmod':leftright_pmod,
        'leftright':leftright,
        'rightleft_pmod':rightleft_pmod,
        'rightleft':rightleft,
        
        'gohigh_pmod':gohigh_pmod,
        'gohigh':gohigh,
        'golow_pmod':golow_pmod,
        'golow':golow,
        
        'gohighleft_pmod':gohighleft_pmod,
        'gohighleft':gohighleft,
        'gohighright_pmod':gohighright_pmod,
        'gohighright':gohighright,
        'golowleft_pmod':golowleft_pmod,
        'golowleft':golowleft,
        'golowright_pmod':golowright_pmod,
        'golowright':golowright,
        
        'highlow_pmod':highlow_pmod,
        'highlow':highlow,
        'lowhigh_pmod':lowhigh_pmod,
        'lowhigh':lowhigh,
        
        'righthighlow_pmod':righthighlow_pmod,
        'lefthighlow_pmod':lefthighlow_pmod,
        'righthighlow':righthighlow,
        'lefthighlow':lefthighlow,
        
        'win':win,
        'loss':loss,
        'winloss':winloss,
        'losswin':losswin,
        
        'sadneutral':sadneutral,
        'neutralsad':neutralsad,
        'sadhappy':sadhappy,
        'happysad':happysad,
        'happyneutral':happyneutral,
        'neutralhappy':neutralhappy,
        'sad':basic_contrasts['Sad'],
        'neutral':basic_contrasts['Neutral'],
        'happy':basic_contrasts['Happy']
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