import numpy

def my_contrasts(design_matrix):
    contrast_matrix = numpy.eye(design_matrix.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])
    
    lefthighlow = basic_contrasts['LeftGo']+basic_contrasts['HighReward']-basic_contrasts['LowReward']
    righthighlow = basic_contrasts['RightGo']+basic_contrasts['HighReward']-basic_contrasts['LowReward']
    gohighlow = basic_contrasts['RightGo']+basic_contrasts['LeftGo']+basic_contrasts['HighReward']-basic_contrasts['LowReward']
    gohigh = basic_contrasts['HighReward'] + basic_contrasts['LeftGo'] + basic_contrasts['RightGo']
    golow = basic_contrasts['LowReward'] + basic_contrasts['LeftGo'] + basic_contrasts['RightGo']
    gohighleft = basic_contrasts['HighReward'] + basic_contrasts['LeftGo']
    gohighright = basic_contrasts['HighReward'] + basic_contrasts['RightGo']
    golowleft = basic_contrasts['LowReward'] + basic_contrasts['LeftGo']
    golowright = basic_contrasts['LowReward'] + basic_contrasts['RightGo']
    highminuslowminusnogo = basic_contrasts['HighReward']-basic_contrasts['LowReward']-basic_contrasts['NoGo']
    
    highlow = basic_contrasts['HighReward'] - basic_contrasts['LowReward']
    lowhigh = basic_contrasts['LowReward'] - basic_contrasts['HighReward']
    
    go = basic_contrasts['RightGo'] + basic_contrasts['LeftGo']
    rightleft = basic_contrasts['RightGo'] - basic_contrasts['LeftGo']
    leftright = basic_contrasts['LeftGo'] - basic_contrasts['RightGo']
    
    sadlowgo = basic_contrasts['Sad'] + basic_contrasts['LowReward'] + basic_contrasts['RightGo'] + basic_contrasts['LeftGo']
    happyhighgo = basic_contrasts['Happy'] + basic_contrasts['HighReward'] + basic_contrasts['RightGo'] + basic_contrasts['LeftGo']
    sadnogo = basic_contrasts['Sad'] + basic_contrasts['NoGo']
    happynogo = basic_contrasts['Happy'] + basic_contrasts['NoGo']
    neutralnogo = basic_contrasts['Neutral'] + basic_contrasts['NoGo']
    
    sadgo = basic_contrasts['Sad'] + basic_contrasts['RightGo'] + basic_contrasts['LeftGo']
    happygo = basic_contrasts['Happy'] + basic_contrasts['RightGo'] + basic_contrasts['LeftGo']
    neutralgo = basic_contrasts['Neutral'] + basic_contrasts['RightGo'] + basic_contrasts['LeftGo']
    
    sadneutral = basic_contrasts['Sad'] - basic_contrasts['Neutral']
    sadhappy = basic_contrasts['Sad'] - basic_contrasts['Happy']
    happyneutral = basic_contrasts['Happy'] - basic_contrasts['Neutral']
    highwin = basic_contrasts['HighWin']
    highloss = basic_contrasts['HighLoss']
    lowwin = basic_contrasts['LowWin']
    lowloss = basic_contrasts['LowLoss']
    win = basic_contrasts['HighWin']+basic_contrasts['LowWin']
    loss = basic_contrasts['LowLoss']+basic_contrasts['HighLoss']
    winloss = basic_contrasts['HighWin']+basic_contrasts['LowWin'] - basic_contrasts['HighLoss'] - basic_contrasts['LowLoss']
    #losswin = basic_contrasts['Loss'] - basic_contrasts['Win']
    gonogo = (basic_contrasts['LeftGo'] + basic_contrasts['RightGo']) - basic_contrasts['NoGo']


    contrasts = {
        'leftgo':basic_contrasts['LeftGo'],
        'rightgo':basic_contrasts['RightGo'],
        'nogo':basic_contrasts['NoGo'],
        
        'happy':basic_contrasts['Happy'],
        'sad':basic_contrasts['Sad'],
        'neutral':basic_contrasts['Neutral'],
        
        'high':basic_contrasts['HighReward'],
        'low':basic_contrasts['LowReward'],
        
        'win':win,
        'loss':loss,
        
        'winloss':winloss,
        'highwin':highwin,
        'highloss':highloss,
        'lowwin':lowwin,
        'lowloss':lowloss,
        #'losswin':losswin,
        'rightleft':rightleft,
        'leftright':leftright,
        'gonogo':gonogo,
        'gohigh':gohigh,
        'golow':golow,
        'gohighleft':gohighleft,
        'gohighright':gohighright,
        'golowleft':golowleft,
        'golowright':golowright,
        'go':go,
        'sadlowgo':sadlowgo,
        'happyhighgo':happyhighgo,
        'sadnogo':sadnogo,
        'happynogo':happynogo,
        'neutralnogo':neutralnogo,
        'sadhappy':sadhappy,
        'sadneutral':sadneutral,
        'happyneutral':happyneutral,
        'sadgo':sadgo,
        'neutralgo':neutralgo,
        'happygo':happygo,
        'highlow':highlow,
        'lowhigh':lowhigh,
        'righthighlow':righthighlow,
        'lefthighlow':lefthighlow,
        'gohighlow':gohighlow,
        'highminuslowminusnogo':highminuslowminusnogo
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
