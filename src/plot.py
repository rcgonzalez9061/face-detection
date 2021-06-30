import os
from src.utils import pickle_read
from PIL import Image, ImageDraw
from IPython.display import display
import glob


PROCESSED_FOLDER = os.path.join('data', 'processed')
TRAIN_IMAGE_FOLDER = os.path.join('data', 'raw', 'WIDER_train', 'images')
VAL_IMAGE_FOLDER = os.path.join('data', 'raw', 'WIDER_val', 'images')

def annotation_generator(bbox):
    blur, expr, illum, inval, occl, pose = bbox[4:]
    return f'{blur = }\n{expr = }\n{illum = }\n{inval = }\n{occl = }\n{pose = }'

class Plotter():
    def __init__(self):
        self.bbx_dict = pickle_read(os.path.join(PROCESSED_FOLDER, 'wider_face_train_bbx_gt.pkl'))
        self.bbx_dict.update(pickle_read(os.path.join(PROCESSED_FOLDER, 'wider_face_val_bbx_gt.pkl')))
        self.train_img_folder = TRAIN_IMAGE_FOLDER
        self.val_img_folder = VAL_IMAGE_FOLDER
    
    def plot_baseline(self, img_name, draw_bbox=False, draw_annotations=False, ipython=True, annot_formatter=annotation_generator):
        img_path = glob.glob(f'data/raw/**/*{img_name}', recursive=True)[0]
        bboxes = self.bbx_dict[img_name]
        annotations = [annot_formatter(bbox) for bbox in bboxes] if draw_annotations else False
        
        
        return self.plot(img_path, bboxes, annotations, ipython)
#         with Image.open(img_path) as img:
#             if draw_bbox:
#                 draw_img = ImageDraw.Draw(img)  
#                 for bbox in bboxes:
#                     bounds = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
#                     draw_img.rectangle(bounds, outline="red", width=2)
#                     if draw_annotations:
#                         draw_img.multiline_text((bbox[0] + bbox[2]  + 3, bbox[1]), )
#             if ipython:
#                 return display(img)
#             else:
#                 return img
    
    def plot(self, img_path, bboxes=None, annotations=False, ipython=True):
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
    
#     def plot_image_jupyter(self, img_name, bbox=False, annotations=False):
#         display(self.plot_image(img_name, bbox, annotations))