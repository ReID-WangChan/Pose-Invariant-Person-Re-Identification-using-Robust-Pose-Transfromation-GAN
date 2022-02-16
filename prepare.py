import os
from shutil import copyfile
from tkinter.tix import Tree
from tqdm import tqdm 

# You only need to change this line to your dataset download path
download_path = 'Datasets/Market-1501-v15.09.15'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/pytorch'
os.makedirs(save_path, exist_ok=True)

#-----------------------------------------
#query
query_path = download_path + '/query'
query_save_path = download_path + '/pytorch/query'
os.makedirs(query_save_path, exist_ok=True)
print(f'query_path = {query_path}')
for root, dirs, files in tqdm(os.walk(query_path, topdown=True)):
    for name in tqdm(files):
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = query_path + '/' + name
        dst_path = query_save_path + '/' + ID[0] 
        os.makedirs(dst_path, exist_ok=True)
        copyfile(src_path, dst_path + '/' + name)

#-----------------------------------------
#multi-query
query_path = download_path + '/gt_bbox'
# for dukemtmc-reid, we do not need multi-query
query_save_path = download_path + '/pytorch/multi-query'
os.makedirs(query_save_path, exist_ok=True)

for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = query_path + '/' + name
        dst_path = query_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

#-----------------------------------------
#gallery
gallery_path = download_path + '/bounding_box_test'
gallery_save_path = download_path + '/pytorch/gallery'
os.makedirs(gallery_save_path, exist_ok=True)

for root, dirs, files in tqdm(os.walk(gallery_path, topdown=True)):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = gallery_path + '/' + name
        dst_path = gallery_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

#---------------------------------------
#train_all
train_path = download_path + '/bounding_box_train'
train_save_path = download_path + '/pytorch/train_all'
os.makedirs(train_save_path, exist_ok=True)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        os.makedirs(dst_path, exist_ok=True)
        copyfile(src_path, dst_path + '/' + name)


#---------------------------------------
#train_val
train_path = download_path + '/bounding_box_train'
train_save_path = download_path + '/pytorch/train'
val_save_path = download_path + '/pytorch/val'
os.makedirs(train_save_path, exist_ok=True)
os.makedirs(val_save_path, exist_ok=True)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.exists(dst_path):
            os.makedirs(dst_path, exist_ok=True)
            dst_path = val_save_path + '/' + ID[0]  #first image is used as val image
            os.makedirs(dst_path, exist_ok=True)
        copyfile(src_path, dst_path + '/' + name)
