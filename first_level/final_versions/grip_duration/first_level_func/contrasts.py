import numpy

def my_contrasts(design_matrix):
    contrast_matrix = numpy.eye(design_matrix.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])
    
    # vigorization
    rightgo = basic_contrasts['RightGoH']+basic_contrasts['RightGoL']
    leftgo = basic_contrasts['LeftGoH']+basic_contrasts['LeftGoL']
    go = basic_contrasts['LeftGoH']+basic_contrasts['LeftGoL']+basic_contrasts['RightGoH']+basic_contrasts['RightGoL']
    gonogo = (basic_contrasts['LeftGoL'] + basic_contrasts['RightGoL'] +
              basic_contrasts['LeftGoH'] + basic_contrasts['RightGoH']) - basic_contrasts['NoGoH'] - basic_contrasts['NoGoL']
    
    # vigor lateralization
    leftright = (basic_contrasts['LeftGoH']+basic_contrasts['LeftGoL'])-(basic_contrasts['RightGoH']+basic_contrasts['RightGoL'])
    rightleft = (basic_contrasts['RightGoH']+basic_contrasts['RightGoL'])-(basic_contrasts['LeftGoH']+basic_contrasts['LeftGoL'])
    
    # inhibition
    nogo = basic_contrasts['NoGoH']+basic_contrasts['NoGoL']
    nogogo = (basic_contrasts['NoGoH'] + basic_contrasts['NoGoL'])-(basic_contrasts['LeftGoL'] + basic_contrasts['RightGoL'] +
              basic_contrasts['LeftGoH'] + basic_contrasts['RightGoH'])
    

    gohigh = basic_contrasts['RightGoH']+basic_contrasts['LeftGoH']
    golow = basic_contrasts['RightGoL']+basic_contrasts['LeftGoL']
    
    gohighleft = basic_contrasts['LeftGoH']
    gohighright = basic_contrasts['RightGoH']
    golowleft = basic_contrasts['LeftGoL']
    golowright = basic_contrasts['RightGoL']

    highlow = (basic_contrasts['RightGoH']+basic_contrasts['LeftGoH'])-(basic_contrasts['RightGoL']+basic_contrasts['LeftGoL'])
    lowhigh = (basic_contrasts['RightGoL']+basic_contrasts['LeftGoL'])-(basic_contrasts['RightGoH']+basic_contrasts['LeftGoH'])
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
        'rightgo':rightgo,
        'leftgo':leftgo,
        'gonogo': gonogo,
        'nogogo':nogogo,
        'nogo':nogo,
        
        'leftright':leftright,
        'rightleft':rightleft,
        
        'gohigh':gohigh,
        'golow':golow,
        
        'gohighleft':gohighleft,
        'gohighright':gohighright,
        'golowleft':golowleft,
        'golowright':golowright,
        
        'highlow':highlow,
        'lowhigh':lowhigh,
        

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