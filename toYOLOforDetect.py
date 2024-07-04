
import cv2
import pandas as pd
import progressbar as pbar

annotations = {}
class_dic = {'pothole':0, 'lateral_crack':1, 'longitudinal_crack': 2,
 'alligator_crack': 3,
 'repair_area': 4,
 'rut': 5}

split = 'val'

file = pd.read_csv('bbox_label_%s.csv'%(split), header=None)

bar = pbar.ProgressBar(max_value=len(file))
for i in range(len(file)):
    dat = file.iloc[i]
    img_name = dat[0]
    x1 = dat[1]
    y1 = dat[2]
    x2 = dat[3]
    y2 = dat[4]
    
    img = cv2.imread('img/%s/%s.jpg'%(split, dat[0]))
    imgh, imgw, _ = img.shape
    
    xnorm = x1/imgw
    ynorm = y1/imgh
    bbox_w = (x2-x1)/imgw
    bbox_h = (y2-y1)/imgh
    
    cname = class_dic[dat[5]]
    
    toAppend = [cname, xnorm, ynorm, bbox_w, bbox_h]
    
    if dat[0] not in list(annotations.keys()):
        annotations[dat[0]] = []
        annotations[dat[0]].append(toAppend)
    else:
        annotations[dat[0]].append(toAppend)
    
    bar.update(i)
    
for k in list(annotations.keys()):
    dat = annotations[k]
    with open('labels/%s/%s.txt'%(split, k), 'w') as f:
        for d in dat:
            f.write('%d %f %f %f %f\n'%(d[0], d[1], d[2], d[3], d[4]))

# img = cv2.imread('img/val/00067826-00003298.jpg')
# cv2.rectangle(img, (1191, 777), (1919, 853), (0,255,0), 2)
# cv2.rectangle(img, (1943, 737), (2029, 748), (0,255,0), 2)
# cv2.rectangle(img, (979, 789), (1132, 815), (0,255,0), 2)
# cv2.imshow('asdf', img)
# cv2.waitKey()
# cv2.destroyAllWindows()