import os
import scipy.io as sio
import matplotlib.pyplot as plt

data_folder = '/mnt/projects/PBCTI/combined/X02441/Task/source_data'

grip_data = sio.loadmat(os.path.join(data_folder,'data_task_X02441_202303241236_run_1.mat'))

force = grip_data['Block_data_left'][0][1][0]
time  = grip_data['Block_time'][0][1][0]
plt.plot(time,force)
plt.show()

force_grip = force[:(len(force) // 2)]
time_grip  = time[:(len(force) // 2)]
plt.plot(time_grip,force_grip)
plt.show()

# before
import numpy
sampling_frq = 120
sample_rt = 1/sampling_frq 
i_time_event = numpy.arange(time_grip[0],time_grip[-1],sample_rt)
i_left_event = numpy.interp(i_time_event,time_grip,force_grip)
slope = numpy.gradient(i_left_event,sample_rt)
plt.plot(i_time_event,slope)
plt.show()

slope = numpy.gradient(i_left_event,i_time_event)
plt.plot(i_time_event,slope)
plt.show()

dF_dt = numpy.gradient(i_left_event) / sample_rt
max(dF_dt)
