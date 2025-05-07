
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Viewer:
    def view_data(self, data, label=""):
        self.start()
        self.add_data(data, label)
        self.end()
    
    def start(self, title="", figure_num=0, xmin=1):
        plt.figure(num=0)
        plt.clf()
        plt.yscale('log')
        plt.xscale('log')
        plt.yticks([0.1,0.05,0.04,0.03,0.02,0.01])
        plt.grid(axis='both', which='both')
        plt.title(title)
        plt.axis([xmin,100,0.005,0.1])
        plt.xlabel("Epochs")
        plt.ylabel("Error %")

    def end(self, block=False, legend_loc="lower left"):
        plt.legend(loc=legend_loc)
        plt.pause(0.5)  # pause a bit so that plots are updated
        plt.show(block=block)

    def add_data(self, data, label="", smooth=[100,10,1]):
        xmin, xmax, ymin, ymax = plt.axis()
        plt.axis([xmin, np.max([xmax,len(data)]), ymin, ymax])
        # if show_unsmoothed:
            # plt.plot(np.arange(1, len(data) + 1), data, label=label)
        for smooth_window in smooth:
            if len(data) >= smooth_window*2:
                window = np.ones(int(smooth_window))/float(smooth_window)
                smoothed = np.convolve(data, window, 'valid')
                plt.plot(np.arange(smooth_window//2, smooth_window//2+len(smoothed)),smoothed, label=label+" "+str(smooth_window))
                break
        
