"""Script to run the outlier detector model
<image_dir> - Directory where images to run the model on are located
<model_path> - Absolute or relative file path to the .pth model file (or the prefix excluding the _0 for an ensemble model)
<output_file> - Absolute or relative file path to where the output dataframe should be written

Usage:
  run_outliers.py <image_dir> <model_path> <output_file> [--gpu=GPU] 
  run_outliers.py (-h | --help)
Examples:
  run_model.py /path/to/images /path/to/model /path/to/write/output.csv
Options:
  -h --help                    Show this screen.
  --gpu=GPU                    Which GPU to use? [Default:None]
"""

#Import packages
import warnings
warnings.simplefilter(action='ignore')
import sys

import os
from docopt import docopt
import pandas as pd

from sklearn.metrics import *

import math
import time

#Define number of cores that the job will use
num_workers = 4

#Image-size to reformat to
size = 320

#Number of output nodes for this model
out_nodes = 1

if __name__ == '__main__':
    arguments = docopt(__doc__)
    
    #Set batch size
    bs = 2
    
    ###Grab image directory
    image_dir = arguments['<image_dir>']
    
    #Set model path 
    mdl_path = arguments['<model_path>']    
    
    #If you are using a GPU-enabled device, specfiy which GPU to use (based on cuda ordering)
    if(arguments['--gpu'] is not None and arguments['--gpu']!="None"):
        os.environ["CUDA_VISIBLE_DEVICES"] = arguments['--gpu']
    
    #Additional import statements after GPU load
    from fastai.callbacks import *
    import fastai
    from fastai.vision import *
    import pretrainedmodels
    import nb_train as nbt

    #Test-time augmentations
    tfms_test = get_transforms(do_flip = False,max_warp = None)
    
    ###Create dummy pandas dataset to store filenames and results
    files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir,f))] 
    
    
    ###Results
    output_df = pd.DataFrame(columns = ['File','Dummy','Prediction'])
    output_df['File'] = files
    output_df['Dummy'] = np.random.random_sample(len(files))
    col = 'Dummy'

    


    #Define a list of images from the specified directory
    imgs = (ImageList.from_df(df=output_df,path=image_dir)
        .split_none()
        .label_from_df(cols=col,label_cls=FloatList)
        .transform(tfms_test,size=size)
        .databunch(num_workers = num_workers,bs=bs).normalize(imagenet_stats))

    #The outlier detector is a Resnet34 CNN
    mdl = fastai.vision.models.resnet34

    #Create model and load weights
    learn = cnn_learner(imgs, mdl)
    learn.model_dir = "."

    learn.load(mdl_path)

    #Define np array to store predictions
    preds = np.zeros(output_df.shape[0])
    learn.model.eval().cuda()
    pbar = progress_bar(range(output_df.shape[0]))
    
    
    
    #Run Inference on each image in the directory after normalizing and resizing to 320x320
    for j in pbar:
        curr = open_image(os.path.join(image_dir,output_df.iloc[j,0]))
        
        
        
        curr = curr.apply_tfms(tfms=tfms_test[0],size=size).px
        img_normalized = curr.clone()

        means = [0.485,0.456,0.406]
        stds = [0.229,0.224,0.225]
        for xx in range(len(means)):
            img_normalized[xx,:,:] = (img_normalized[xx,:,:] - means[xx]) / stds[xx]

        out = learn.model(img_normalized.view(1,3,size,size).cuda())
        preds[j] = out.cpu().detach().data

    #Save predictions back to the dataframe
    output_df['Outlier_Score'] = preds
    output_df = output_df.drop(["Dummy","Prediction"],axis=1).reset_index(drop=True)

    output_df['Outlier_Flag'] = [1 if x > 0.45 else 0 for x in output_df['Outlier_Score']]
    
    #Output to file
    output_df.to_csv(arguments['<output_file>'],index=False)


