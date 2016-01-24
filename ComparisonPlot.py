import matplotlib
import matplotlib.rcsetup as rcsetup

matplotlib.use('WXAgg')
print(rcsetup.all_backends)
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class ComparisonPlot():
    def __init__(self, plot_queue):
        # super(ComparisonPlot, self).__init__()

        self.plot_queue = plot_queue
        self.ax1 = None
        self.ani = None

        self.run()
        # self.start()

    def run(self):
        def animate(i, xy_queue, xy):
            print xy_queue.qsize()
            if xy_queue.qsize() > 1:
                for i in range(0, xy_queue.qsize()):
                    t = xy_queue.get()
                    xy[0].append(t[0])
                    xy[1].append(t[1])

            self.ax1.clear()
            self.ax1.plot(xy[0], xy[1])

        fig = plt.figure()
        self.ax1 = fig.add_subplot(1, 1, 1)
        self.ani = animation.FuncAnimation(fig, animate, interval=1000, fargs=(self.plot_queue, [[], []]))
        plt.show()
