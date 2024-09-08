import pandas as pd
import matplotlib.pyplot as plt
import os

working_dir='/mnt/projects/PBCTI/combined/Results'
beh_old_values = pd.read_excel(os.path.join(working_dir,'Group_output_202402081541/behavioral_group_output_withreject.xlsx'))
beh_new_values = pd.read_excel(os.path.join(working_dir,'Group_output_202406131123/behavioral_group_output_withreject.xlsx'))


plt.scatter(beh_old_values['Slope'],beh_new_values['Slope'])
plt.xlabel('Old values')
plt.ylabel('New values')
plt.show()

std_old = beh_old_values['Slope'].std()
std_new = beh_new_values['Slope'].std()
