import matplotlib
import matplotlib.pyplot as plt
from .image_utils import denormalize_img_tensor
from .colorizer import Colorizer
from matplotlib.patches import Rectangle
from IPython import display as idisplay


class SegmentationVisualizer:

    def __init__(self, im_norm=True):
        plt.ion()
        self.inline = 'inline' in matplotlib.get_backend()
        self.dh = None
        self.im_norm = im_norm
        self.colorizer = Colorizer()
        self.window = plt.subplots(ncols=3)

    def display(self, image, pred, gt):
        fig, axs = self.window
        for ax in axs:
            ax.cla()
            ax.set_axis_off()

        # display image
        if self.im_norm:
            image = image.squeeze_()
            image = denormalize_img_tensor(image)
        
        axs[0].imshow(image.numpy().transpose(1,2,0))

        # display pred
        colored_pred = self.colorizer.colorize(pred)
        axs[1].imshow(colored_pred.numpy().transpose(1,2,0))

        # display gt
        colored_gt = self.colorizer.colorize(gt)
        axs[2].imshow(colored_gt.numpy().transpose(1,2,0))

        if self.inline: 
            if self.dh:
                self.dh.update(fig)
            else:
                self.dh = idisplay.display(fig, display_id=True)
        else:
            plt.draw()
        
        self.mypause(0.001)

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return




