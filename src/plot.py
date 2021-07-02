import os, glob
from src.utils import pickle_load
from PIL import Image, ImageDraw
from IPython.display import display

PROCESSED_FOLDER = os.path.join('data', 'processed')
TRAIN_IMAGE_FOLDER = os.path.join('data', 'raw', 'WIDER_train', 'images')
VAL_IMAGE_FOLDER = os.path.join('data', 'raw', 'WIDER_val', 'images')

def annotation_generator(bbox):
    '''Returns the default annotation string for a face's bounding box.'''
    blur, expr, illum, inval, occl, pose = bbox[4:]
    return f'{blur = }\n{expr = }\n{illum = }\n{inval = }\n{occl = }\n{pose = }'

class Plotter():
    '''Image plotter to assist with analysis and display of images from the WDIER dataset.'''
    def __init__(self):
        self.bbx_dict = pickle_load(os.path.join(PROCESSED_FOLDER, 'wider_face_train_bbx_gt.pkl'))
        self.bbx_dict.update(pickle_load(os.path.join(PROCESSED_FOLDER, 'wider_face_val_bbx_gt.pkl')))
        self.train_img_folder = TRAIN_IMAGE_FOLDER
        self.val_img_folder = VAL_IMAGE_FOLDER
    
    def display_baseline(self, img_name, draw_bbox=False, draw_annotations=False, ipython=True, annot_formatter=annotation_generator):
        '''
        Plots the baselines of a given image in the training or validation sets.
        
        img_name: File name of the image to be plotted. (Note: Do not use image path. It is not necessary)
        draw_bbox: If true, will overlay bounding boxes for all faces according to the baseline.
        draw_annotations: If true and draw bbox is true, bboxes will also be drawn with their annotation.
        ipython: If true, use IPython display output.
        annot_formatter (func): Given a bbox, returns the formated string representation for annotations.
        '''
        img_path = glob.glob(f'data/raw/**/*{img_name}', recursive=True)[0]
        bboxes = self.bbx_dict[img_name]
        annotations = [annot_formatter(bbox) for bbox in bboxes] if draw_annotations else False
        
        
        return self.display(img_path, bboxes, annotations, ipython)
    
    def display(self, img_path, bboxes=None, annotations=False, ipython=True):
        '''
        Displays the given image.
        
        draw_bbox (List): If provided, will overlay bounding boxes for all faces.
        annotations (List): List of annotations (one per bbox). If provided, will draw bboxes with annotations.
        ipython: If true, use IPython display output.
        '''
        with Image.open(img_path) as img:
            if bboxes:
                draw_img = ImageDraw.Draw(img)  
                for idx, bbox in enumerate(bboxes):
                    bounds = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
                    draw_img.rectangle(bounds, outline="red", width=2)
                    if annotations:
                        draw_img.multiline_text((bbox[0] + bbox[2]  + 3, bbox[1]), annotations[idx])
            if ipython:
                return display(img)
            else:
                return img