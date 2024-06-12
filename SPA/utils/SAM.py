from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def show_anns(anns, show_bbox=True,region_nums=10):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    color_lt = []
    # if len(sorted_anns) < region_nums -1:
    #     region_nums = len(sorted_anns) - 1
    # for ann in sorted_anns[1:region_nums+1]:
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [1]])
        color_lt.append(color_mask)
        img[m] = color_mask
        
    if show_bbox:
        for ann in sorted_anns:
            x , y, w, h = ann['bbox']
            
            rect = plt.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)

    ax.imshow(img)
    return color_lt
        

class sam_controller():
    def __init__(self, model_path, model_type, device, \
                samParas={'points_per_side':64,
                          'pred_iou_thresh':0.95,
                          'stability_score_thresh':0.95,
                          'crop_n_layers':1,
                          'crop_n_points_downscale_factor':2,
                          'min_mask_region_area':200,
                          'region_nums':11}) -> None:
        
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=self.model_path)
        self.samParas = samParas
        self.sam.to(device=device)
        
    def loda_img(self, img_path):
        #loda path to cv2 img
        self.img = cv.imread(img_path)
        print("loading img:", img_path)
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)

    def generate_mask(self, img_path):
        self.loda_img(img_path=img_path)
        mask_generator= SamAutomaticMaskGenerator(
                                                    model=self.sam,
                                                    points_per_side=self.samParas['points_per_side'],
                                                    pred_iou_thresh=self.samParas['pred_iou_thresh'],
                                                    stability_score_thresh=self.samParas['stability_score_thresh'],
                                                    crop_n_layers=self.samParas['crop_n_layers'],
                                                    crop_n_points_downscale_factor=self.samParas['crop_n_points_downscale_factor'],
                                                    min_mask_region_area=self.samParas['min_mask_region_area']  # Requires open-cv to run post-processing
                                                )
        #mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.masks = mask_generator.generate(self.img)
        
        return self.masks
    
    def plot_mask(self, output_path, show_bbox=True):
        plt.figure(figsize=(20,20))
        plt.imshow(self.img)
        region_nums = self.samParas['region_nums']
        color_lt = show_anns(self.masks, show_bbox=show_bbox, region_nums=region_nums)
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
        return color_lt
        
