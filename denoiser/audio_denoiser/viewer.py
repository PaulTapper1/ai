
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
        plt.yticks([0.5,0.4,0.3,0.2,0.1,0.05,0.04,0.03,0.02,0.01])
        plt.grid(axis='both', which='both')
        plt.title(title)
        plt.axis([xmin,100,0.01,0.5])
        plt.xlabel("Epochs")
        plt.ylabel("Error %")

    def end(self, block=False, legend_loc="lower left"):
        plt.legend(loc=legend_loc)
        plt.pause(0.2)  # pause a bit so that plots are updated
        plt.show(block=block)

    def add_data(self, data, label="", smooth=10, show_unsmoothed=True):
        xmin, xmax, ymin, ymax = plt.axis()
        plt.axis([xmin, np.max([xmax,len(data)]), ymin, ymax])
        if show_unsmoothed:
            plt.plot(data, label=label)
        if smooth>0 and len(data) >= smooth:
            window = np.ones(int(smooth))/float(smooth)
            smoothed = np.convolve(data, window, 'valid')
            plt.plot(np.arange(smooth//2, smooth//2+len(smoothed)),smoothed, label=label+" smoothed")
        
