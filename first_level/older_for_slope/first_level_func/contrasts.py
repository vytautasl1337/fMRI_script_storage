import numpy

def my_contrasts(design_matrix):
    contrast_matrix = numpy.eye(design_matrix.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])
    
    #lefthighlow = basic_contrasts['LeftGo']+basic_contrasts['HighReward']-basic_contrasts['LowReward']
    #righthighlow = basic_contrasts['RightGo']+basic_contrasts['HighReward']-basic_contrasts['LowReward']
    #gohighlow = basic_contrasts['RightGo']+basic_contrasts['LeftGo']+basic_contrasts['HighReward']-basic_contrasts['LowReward']
    gohigh = basic_contrasts['LeftGoH'] + basic_contrasts['RightGoH']
    golow = basic_contrasts['LeftGoL'] + basic_contrasts['RightGoL']
    gohighleft = basic_contrasts['LeftGoH']
    gohighright = basic_contrasts['RightGoH']
    golowleft = basic_contrasts['LeftGoL']
    golowright = basic_contrasts['RightGoL']
    gohighgolow = basic_contrasts['RightGoH']+basic_contrasts['LeftGoH']-basic_contrasts['RightGoL']-basic_contrasts['LeftGoL']
    win = basic_contrasts['Win']
    loss = basic_contrasts['Loss']
    # highminuslowminusnogo = basic_contrasts['HighReward']-basic_contrasts['LowReward']-basic_contrasts['NoGo']
    
    # highlow = basic_contrasts['HighReward'] - basic_contrasts['LowReward']
    # lowhigh = basic_contrasts['LowReward'] - basic_contrasts['HighReward']
    
    # go = basic_contrasts['RightGo'] + basic_contrasts['LeftGo']
    # rightleft = basic_contrasts['RightGo'] - basic_contrasts['LeftGo']
    # leftright = basic_contrasts['LeftGo'] - basic_contrasts['RightGo']
    
    # sadlowgo = basic_contrasts['Sad'] + basic_contrasts['LowReward'] + basic_contrasts['RightGo'] + basic_contrasts['LeftGo']
    # happyhighgo = basic_contrasts['Happy'] + basic_contrasts['HighReward'] + basic_contrasts['RightGo'] + basic_contrasts['LeftGo']
    # sadnogo = basic_contrasts['Sad'] + basic_contrasts['NoGo']
    # happynogo = basic_contrasts['Happy'] + basic_contrasts['NoGo']
    # neutralnogo = basic_contrasts['Neutral'] + basic_contrasts['NoGo']
    
    # sadgo = basic_contrasts['Sad'] + basic_contrasts['RightGo'] + basic_contrasts['LeftGo']
    # happygo = basic_contrasts['Happy'] + basic_contrasts['RightGo'] + basic_contrasts['LeftGo']
    # neutralgo = basic_contrasts['Neutral'] + basic_contrasts['RightGo'] + basic_contrasts['LeftGo']
    
    sadneutral = basic_contrasts['Sad'] - basic_contrasts['Neutral']
    sadhappy = basic_contrasts['Sad'] - basic_contrasts['Happy']
    happyneutral = basic_contrasts['Happy'] - basic_contrasts['Neutral']
    winloss = basic_contrasts['Win'] - basic_contrasts['Loss']
    # losswin = basic_contrasts['Loss'] - basic_contrasts['Win']
    gonogo = (basic_contrasts['LeftGoL'] + basic_contrasts['RightGoL'] +
              basic_contrasts['LeftGoH'] + basic_contrasts['RightGoH']) - basic_contrasts['NoGoH'] - basic_contrasts['NoGoL']


    contrasts = {
        # 'leftgo':basic_contrasts['LeftGo'],
        # 'rightgo':basic_contrasts['RightGo'],
        # 'nogo':basic_contrasts['NoGo'],
        
        'happy':basic_contrasts['Happy'],
        'sad':basic_contrasts['Sad'],
        'neutral':basic_contrasts['Neutral'],
        
        # 'high':basic_contrasts['HighReward'],
        # 'low':basic_contrasts['LowReward'],
        
        # 'win':basic_contrasts['Win'],
        # 'loss':basic_contrasts['Loss'],
        
        'winloss':winloss,
        # 'losswin':losswin,
        # 'rightleft':rightleft,
        # 'leftright':leftright,
        'gonogo':gonogo,
        'gohigh':gohigh,
        'golow':golow,
        'gohighleft':gohighleft,
        'gohighright':gohighright,
        'golowleft':golowleft,
        'golowright':golowright,
        'gohighgolow':gohighgolow,
        'win':win,
        'loss':loss,
        # 'go':go,
        # 'sadlowgo':sadlowgo,
        # 'happyhighgo':happyhighgo,
        # 'sadnogo':sadnogo,
        # 'happynogo':happynogo,
        # 'neutralnogo':neutralnogo,
        'sadhappy':sadhappy,
        'sadneutral':sadneutral,
        'happyneutral':happyneutral,
        # 'sadgo':sadgo,
        # 'neutralgo':neutralgo,
        # 'happygo':happygo,
        # 'highlow':highlow,
        # 'lowhigh':lowhigh,
        # 'righthighlow':righthighlow,
        # 'lefthighlow':lefthighlow,
        # 'gohighlow':gohighlow,
        # 'highminuslowminusnogo':highminuslowminusnogo
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