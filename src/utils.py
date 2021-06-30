import re, pickle, glob, os

def pickle_save(obj, path):
    try:
        with open(path, 'wb') as file:
            pickle.dump(obj, file)
            return True
    except Error as e:
        raise e
        
def pickle_read(path):
    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise e
        
def extract_bbox(bbox_fp):
    '''
    Load the bboxes from their files into a dictionary indexed by image name.
    
    bbox_fp: file path to the bbox txt file.
    '''
    bbox_dict = {}
    with open(bbox_fp, 'r') as bbox_file:
        current_image = None
        current_bboxes = None
        num_bboxes = None
        
        for line in bbox_file:
            if re.match(r'^\d+--\w+\/\d+_[\w_]+.jpg', line):
                if current_image is not None:
                    bbox_dict[current_image] = current_bboxes
                current_image = line.strip()
                current_bboxes = []
            elif re.match(r'^\d+$', line):
                num_bboxes = int(line.strip())
            elif re.match(r'^\d+ \d+ \d+ \d+ \d+ \d+ \d+ \d+ \d+ \d+', line):
                current_bboxes.append([int(n) for n in line.split()])
                
    return bbox_dict
        
def convert_bbx_gt_to_pickle(folder):
    for path in glob.glob(os.path.join(folder, '*bbx_gt.txt')):
        file_name = os.path.split(path)[-1].replace('.txt', '')
        outpath = os.path.join('data', 'processed', f'{file_name}.pkl')
        pickle_save(extract_bbox(path), outpath)


