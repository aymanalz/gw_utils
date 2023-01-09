import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import path
from matplotlib.widgets import Button

def main():
    arr = np.random.randn(200,100)
    zone =Zone_selector(arr)
    #plt.imshow(zone.mask)
    plt.show()
    pass

global polygx, ploygy, ax, poly, ppath, flg
polygx = []
ploygy = []
ppath = []
flg = 0

class Zone_selector(object):
    def __init__(self, arr, fgax = []):
        self.arr = arr
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(arr)


        if len(fgax)> 0:
            try:
                plt.scatter(fgax[:, 1], fgax[:, 0], c=fgax[:, 2], cmap='jet')
                plt.colorbar()
                #plt.contour(fgax[2],50)
                #plt.imshow(fgax[3], alpha=0.3)
            except:
                pass


        plt.subplots_adjust(bottom=0.2)
        cid1 = self.fig.canvas.mpl_connect('button_press_event', lambda event: self.onclick2(event))
        cid2 = self.fig.canvas.mpl_connect('close_event', lambda event: self.cclose2(event))

        # add butons
        axclose = plt.axes([0.7, 0.04, 0.1, 0.075])
        bnext = Button(axclose, 'Close')
        bnext.on_clicked(self.cclose2)

        plt.show()

    def plot_cs(self, event):
        pass


    def cclose(self, event):
        p = path.Path(ppath)
        nrows, ncols = self.arr.shape
        x = np.linspace(0,ncols,ncols)
        y = np.linspace(0,nrows,nrows)
        xx, yy = np.meshgrid(x,y)
        xx = xx.flatten()
        yy = yy.flatten()
        points = np.vstack((yy, xx)).transpose()
        mask = p.contains_points(points)
        self.mask = mask.reshape(nrows, ncols)

    def cclose2(self, event):
        plt.close()
        plt.figure()
        flg = 1
        self.polygx = polygx
        self.ploygy = ploygy

        return polygx,ploygy



    def onclick(self, event ):
        ploygy.append(event.ydata)
        polygx.append(event.xdata)
        ppath.append((event.ydata,event.xdata ))
        try:
            self.poly.remove()
        except:
            pass
        #polygx.append(polygx[0])
        #ploygy.append(ploygy[0])
        self.ax.scatter(polygx,ploygy, color='r')
        xy = np.array([polygx, ploygy])
        p1 = Polygon(xy.transpose(), True, edgecolor='k', facecolor = 'y' , alpha = 0.5)
        self.poly = self.ax.add_patch(p1)
        event.canvas.draw()
        event.canvas.flush_events()
        plt.show()
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
    def onclick2(self, event ):
        if flg == 0:
            ploygy.append(event.ydata)
            polygx.append(event.xdata)
            ppath.append((event.ydata,event.xdata ))
            try:
                self.poly.remove()
            except:
                pass
            #polygx.append(polygx[0])
            #ploygy.append(ploygy[0])
            self.ax.scatter(polygx,ploygy, color='r')
            self.ax.plot(polygx,ploygy, color='k')
            #xy = np.array([polygx, ploygy])
            #p1 = Polygon(xy.transpose(), True, edgecolor='k', facecolor = 'y' , alpha = 0.5)
            #self.poly = self.ax.add_patch(p1)
            event.canvas.draw()
            event.canvas.flush_events()
            plt.show()
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))




if __name__ == "__main__":
    #main()
    pass

