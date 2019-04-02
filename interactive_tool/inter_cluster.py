from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import skimage.io
import numpy as np
from keras import Model
from tkinter import *
from PIL import ImageTk, Image

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import skimage.io

from models.UNetValid import get_model
from utils import clustering

img_path = '/home/zhygallo/zhygallo/zeiss/clusternet_segmentation/data/crop_img.png'


class Paint(object):

    def __init__(self):
        self.root = Tk()
        self.mask_clust = Toplevel()

        self.img_np = skimage.io.imread(img_path)
        img = Image.fromarray(self.img_np)
        self.img = ImageTk.PhotoImage(image=img)
        frame = Frame(self.root, width=self.img.width(), height=self.img.height())
        frame.grid(row=1, columnspan=2)
        self.canvas = Canvas(frame, bg='#FFFFFF', width=512, height=512,
                             scrollregion=(0, 0, self.img.width(), self.img.height()))
        hbar = Scrollbar(frame, orient=HORIZONTAL)
        hbar.pack(side=BOTTOM, fill=X)
        hbar.config(command=self.canvas.xview)
        vbar = Scrollbar(frame, orient=VERTICAL)
        vbar.pack(side=RIGHT, fill=Y)
        vbar.config(command=self.canvas.yview)
        self.canvas.config(width=512, height=512)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.pack(side=LEFT, expand=True, fill=BOTH)

        # self.c = Canvas(self.root, bg='black', width=self.img.width(), height=self.img.height())
        # self.c.grid(row=1, columnspan=5)

        self.done_button = Button(self.root, text='DONE', command=self.done)
        self.done_button.grid(row=0, column=1)

        self.pos_stroke = BooleanVar()
        self.stroke_type_button = Checkbutton(self.root, text='positive', variable=self.pos_stroke)
        self.stroke_type_button.grid(row=0, column=0)

        self.fig_mask_clust = Figure(figsize=(4, 4))
        self.subplot_mask = self.fig_mask_clust.add_subplot(111)
        self.subplot_mask.axis('off')

        self.mask_clust_c = FigureCanvasTkAgg(self.fig_mask_clust, master=self.mask_clust)
        self.mask_clust_c.show()
        self.mask_clust_c.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None

        self.num_clusters = 100

        model_weights_path = '/home/zhygallo/zhygallo/zeiss/clusternet_segmentation/data/results/38/model.h5'
        img_path = '/home/zhygallo/zhygallo/zeiss/clusternet_segmentation/data/raw/test/images/JNCASR_Overall_Inducer_123_Folder_F_30_ebss_07_R3D_D3D_PRJ.png'
        self.image = skimage.io.imread(img_path).astype('float32')

        self.image = np.expand_dims(self.image, -1)
        # mean = np.array([9.513245, 10.786118, 10.060172], dtype=np.float32).reshape(1, 1, 3)
        # std = np.array([18.76071, 19.97462, 19.616455], dtype=np.float32).reshape(1, 1, 3)
        mean = np.array([30.17722], dtype='float32').reshape(1, 1, 1)
        std = np.array([33.690296], dtype='float32').reshape(1, 1, 1)
        self.image -= mean
        self.image /= std
        # self.image /= 255

        self.model = get_model(self.image.shape, self.num_clusters)
        self.model.load_weights(model_weights_path)

        before_last_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer('for_clust').output)

        self.x = before_last_layer_model.predict(np.expand_dims(self.image, axis=0), verbose=1, batch_size=1)

        self.pos_neg_mask = np.zeros((self.x.shape[1], self.x.shape[2]))
        # self.pos_neg_mask = np.load(
        #     '/home/zhygallo/zhygallo/inria/cluster_seg/deepcluster_segmentation/data/patches_512_lux/pos_neg_mask.npy')
        self.x_flat = np.reshape(self.x, (self.x.shape[0] * self.x.shape[1] * self.x.shape[2], self.x.shape[-1]))

        self.n_pix, self.dim_pix = self.x_flat.shape[0], self.x_flat.shape[1]

        self.deepcluster = clustering.Kmeans(self.num_clusters)
        self.centroids = self.deepcluster.cluster(self.x_flat, verbose=True, init_cents=None)

        masks_flat = np.zeros((self.n_pix, 1))
        for clust_ind, clust in enumerate(self.deepcluster.images_lists):
            masks_flat[clust] = clust_ind
        self.masks = masks_flat.reshape((self.x.shape[1], self.x.shape[2]))

        self.count = 1
        skimage.io.imsave(
            '/home/zhygallo/zhygallo/zeiss/clusternet_segmentation/data/binar_pred/clust_%i.png' % self.count,
            self.masks.astype('uint32'))
        self.subplot_mask.imshow(self.masks)
        self.canvas.create_image(0, 0, image=self.img, anchor=NW)

        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

    def activate_button(self, some_button):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button

    def paint(self, event):
        self.line_width = 3.0
        if self.pos_stroke.get():
            self.color = 'blue'
        else:
            self.color = 'red'
        paint_color = self.color
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y,
                                    self.canvas.canvasx(event.x), self.canvas.canvasy(event.y),
                                    width=self.line_width, fill=paint_color,
                                    capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = self.canvas.canvasx(event.x)
        self.old_y = self.canvas.canvasy(event.y)
        if self.pos_stroke.get():
            self.pos_neg_mask[int(self.canvas.canvasy(event.y)), int(self.canvas.canvasx(event.x))] = 1
        else:
            self.pos_neg_mask[int(self.canvas.canvasy(event.y)), int(self.canvas.canvasx(event.x))] = -1

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def done(self):
        np.save('/home/zhygallo/zhygallo/zeiss/clusternet_segmentation/data/pos_neg_mask_transfer.npy',
                self.pos_neg_mask)
        pos_clusts = np.unique(self.masks[self.pos_neg_mask == 1])
        neg_clusts = np.unique(self.masks[self.pos_neg_mask == -1])

        intersect = np.intersect1d(pos_clusts, neg_clusts)

        for clust in intersect:
            pos = np.sum((self.masks == clust) * (self.pos_neg_mask == 1))
            neg = np.sum((self.masks == clust) * (self.pos_neg_mask == -1))
            pos_ratio = pos / (pos + neg)
            neg_ratio = neg / (pos + neg)

            if pos_ratio < 0.3 or neg_ratio < 0.3:
                continue

            self.centroids = np.delete(self.centroids, clust, axis=0)
            neg_centroid = self.x[0][((self.masks == clust) *
                                      (self.pos_neg_mask == -1))].mean(axis=0).reshape(1, self.centroids.shape[1])
            pos_centroid = self.x[0][((self.masks == clust) *
                                      (self.pos_neg_mask == 1))].mean(axis=0).reshape(1, self.centroids.shape[1])

            self.centroids = np.concatenate((self.centroids, neg_centroid, pos_centroid), axis=0)

        self.deepcluster = clustering.Kmeans(self.centroids.shape[0])
        self.centroids = self.deepcluster.cluster(self.x_flat, verbose=True, init_cents=self.centroids)

        masks_flat = np.zeros((self.n_pix, 1))
        for clust_ind, clust in enumerate(self.deepcluster.images_lists):
            masks_flat[clust] = clust_ind
        self.masks = masks_flat.reshape((self.x.shape[1], self.x.shape[2]))

        pos_clusts = np.unique(self.masks[self.pos_neg_mask == 1])
        neg_clusts = np.unique(self.masks[self.pos_neg_mask == -1])

        intersect = np.intersect1d(pos_clusts, neg_clusts)

        pos_neg_arr = np.zeros(self.masks.shape)
        pos_neg_arr[np.isin(self.masks, pos_clusts)] = 1
        pos_neg_arr[np.isin(self.masks, neg_clusts)] = 2
        pos_neg_arr[np.isin(self.masks, intersect)] = 3
        self.subplot_mask.imshow(pos_neg_arr)

        self.mask_clust_c.show()

        pos_arr = np.zeros(self.masks.shape)
        pos_arr[np.isin(self.masks, pos_clusts)] = 1
        for clust in intersect:
            pos = np.sum((self.masks == clust) * (self.pos_neg_mask == 1))
            neg = np.sum((self.masks == clust) * (self.pos_neg_mask == -1))
            pos_ratio = pos / (pos + neg)

            if pos_ratio < 0.3:
                pos_arr[self.masks == clust] = 0

        # np.save('/home/zhygallo/zhygallo/zeiss/clusternet_segmentation/data/pos_arr.npy', pos_clusts)

        skimage.io.imsave(
            '/home/zhygallo/zhygallo/zeiss/clusternet_segmentation/data/binar_pred/pos_%i.png' % self.count,
            np.isin(self.masks, pos_clusts))
        skimage.io.imsave(
            '/home/zhygallo/zhygallo/zeiss/clusternet_segmentation/data/binar_pred/only_pos_%i.png' % self.count,
            np.isin(self.masks, pos_clusts) * (1 - np.isin(self.masks, neg_clusts)))
        skimage.io.imsave(
            '/home/zhygallo/zhygallo/zeiss/clusternet_segmentation/data/binar_pred/filt_pos_%i.png' % self.count,
            pos_arr.astype('uint16'))
        self.count += 1


def main():
    Paint()


if __name__ == "__main__":
    main()
