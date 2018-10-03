
# coding: utf-8

# In[ ]:


import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
from PIL import Image

get_ipython().run_line_magic('matplotlib', 'inline')

from shutil import copyfile


# In[ ]:


# Look at a few examples:
loc = 'C:\\Users\\HIMANSHU\\Desktop\\CSE Project\\BreaKHis_v1'
all_image_locs = glob.glob(loc +'\\**\\*.png', recursive=True)

classes = ['B_A', 'B_F', 'B_PT', 'B_TA', 'M_DC', 'M_LC', 'M_MC', 'M_PC']

for slide_class in classes:
    image_locs = [loc for loc in all_image_locs if loc.rsplit('\\', 1)[1].split('_', 1)[1].split('-', 1)[0] == slide_class]
    image_locs = np.random.choice(image_locs, 2)

    f = plt.figure(figsize=(11,8))
    for i in range(len(image_locs)):
        sp = f.add_subplot(2, len(image_locs)//1, i+1)
        sp.axis('Off')
        sp.set_title(image_locs[i].rsplit('\\', 1)[-1], fontsize=10)
        image = np.asarray(Image.open(image_locs[i]))
        plt.title(slide_class)

        #plt.tight_layout()
        plt.imshow(image)


# In[ ]:


#make test train and valid sets
def train_test_valid_split(current_dir, out_dir, valid_proportion, test_proportion):
    # Make directories to store files        
    train_dir=os.path.join(out_dir, 'train')
    valid_dir=os.path.join(out_dir, 'valid')
    test_dir=os.path.join(out_dir, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    all_files = glob.glob(current_dir+'/**/*.png', recursive=True)
    all_files = [loc for loc in all_files if loc.split('.', 1)[-1] == 'png']
    print('len(all_files): ', len(all_files))
    
    # get all the patient ids
    all_ids = [loc.rsplit('/', 1)[1].split('-', 3)[2] for loc in all_files]
    all_ids = list(set(all_ids))
    print('Number of Patients: ',  len(all_ids))

    # Get all the tumor classes, and stratify the patients so test and valid sets contain at least one from each type
    tumor_class_list = list(set([loc.rsplit('/', 1)[1].split('_', 1)[1].split('-', 1)[0] for loc in all_files]))

    for tumor_class in tumor_class_list:
        # for each tumor class find all the patients, and randomly split up the patients into test, train, valid
        class_ids = [loc.rsplit('/', 1)[1].split('-', 3)[2] for loc in all_files if loc.rsplit('/', 1)[1].split('_', 1)[1].split('-', 1)[0] == tumor_class]
        random.shuffle(class_ids)
        class_ids = list(set(class_ids))
        num_ids = len(class_ids)
        print('Number of Patients of type ',  tumor_class, ' is: ', num_ids)

        valid_ids = list(set(class_ids[0:int(np.ceil(num_ids*valid_proportion))]))
        test_ids = list(set(class_ids[int(np.ceil(num_ids*valid_proportion)) : int(np.ceil(num_ids*(valid_proportion+test_proportion)))]))
        train_ids = list(set(class_ids[int(np.ceil(num_ids*(valid_proportion+test_proportion))):]))

        print('len(train_ids)', len(train_ids))
        print('len(test_ids)', len(test_ids))
        print('len(valid_ids)', len(valid_ids))

        train_files = [loc for loc in all_files if loc.rsplit('/', 1)[1].split('-', 3)[2] in train_ids]
        test_files = [loc for loc in all_files if loc.rsplit('/', 1)[1].split('-', 3)[2] in test_ids]
        valid_files = [loc for loc in all_files if loc.rsplit('/', 1)[1].split('-', 3)[2] in valid_ids]

        print('len(train_files)', len(train_files))
        print('len(test_files)', len(test_files))
        print('len(valid_files)', len(valid_files))

        # for each of train, test, valid make directories for each class, and copy the 
        for file in train_files:
            name = file.rsplit('/', 1)[1]
            tumor_class = file.rsplit('/', 1)[1].split('_', 1)[1].split('-', 1)[0]
            new_folder = os.path.join(train_dir, tumor_class)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            copyfile(file, os.path.join(new_folder, name))

        for file in valid_files:
            name = file.rsplit('/', 1)[1]
            tumor_class = file.rsplit('/', 1)[1].split('_', 1)[1].split('-', 1)[0]
            new_folder = os.path.join(valid_dir, tumor_class)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            copyfile(file, os.path.join(new_folder, name))

        for file in test_files:
            name = file.rsplit('/', 1)[1]
            tumor_class = file.rsplit('/', 1)[1].split('_', 1)[1].split('-', 1)[0]
            new_folder = os.path.join(test_dir, tumor_class)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            copyfile(file, os.path.join(new_folder, name))


# In[ ]:


# create the main sets:
current_dir = 'C:\\Users\\HIMANSHU\\Desktop\\CSE Project\\BreaKHis_v1'
out_dir = 'C:\\Users\\HIMANSHU\\Desktop\\CSE Project\\BreaKHis_v1\\by_patient'
valid_proportion = .2
test_proportion = .2

# train_test_valid_split(current_dir, out_dir, valid_proportion, test_proportion)

