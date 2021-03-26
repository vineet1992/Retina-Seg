"""This provides auxillary functions for training a combined autoencoder/predictor """

from fastai import *
from fastai.tabular import *
from fastai.vision import *
import fastai.data_block
import torchvision
import os
from torchvision.utils import save_image

###This is future code for how to implement the loss function, think the best move here will be to ignore fastai and just do this in Tensor land
#model.zero_grad()
#output = model(x)
#loss = criterion(output, target)
#loss = loss + torch.norm(model[0].weight)
#loss.backward()

###TODO Try buffering images in memory, might significantly improve speed (if this is feasible)
class SegmentationDataset(Dataset):
    """A Dataset of combined image names, and targets."""
    ##x_img = ImageList
    ##y = Any way of representing targets that is accessible via indexing
    def __init__(self, fnames,img_dir,mask_dir,transforms=None,size=None):
        self.img_dir, self.fnames,self.mask_dir = img_dir,fnames,mask_dir
        self.is_empty = False
        self.transforms = transforms
        self.null_tfms = get_transforms(do_flip = False,max_warp = None)
        self.size = size
        self.c = 0
        self.num_outputs = 1

    def __len__(self): return len(self.fnames)


    def __getitem__(self, i):
        img = open_image(os.path.join(self.img_dir,self.fnames.iloc[i]))
        target_img = open_image(os.path.join(self.mask_dir,self.fnames.iloc[i]))
        #import pdb; pdb.set_trace()
        if(self.transforms is not None):
            if(self.size is not None):
                img = img.apply_tfms(tfms=self.transforms[0],size=self.size).px
                target_img = target_img.apply_tfms(tfms=self.transforms[0],size=self.size,do_resolve=False).px
            else:
                img = img.apply_tfms(tfms=self.transforms[0]).px
                target_img = target_img.apply_tfms(tfms=self.transforms[0],do_resolve=False).px
        else:
            if(self.size is not None):
                img = img.resize(self.size).px
                target_img = target_img.resize(self.size).px
        #target_img = torch.ceil(target_img)
        
        target_img = torch.round(target_img)
        return img,target_img


class SegmentationModel(nn.Module):
    """This model produces a reconstructed image along with a prediction"""
    def __init__(self, encoder,decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.gradients = None
        self.encoding = None
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    
    def forward(self, *x):
        encoding = self.encoder(x[0])
        img = self.decoder(encoding)
        return img

class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, output, mask):
        num = torch.mul(output,mask).sum()
        denom = output.sum() + mask.sum()
        dice = 2*num / denom
        return dice

class CrossEntropy(nn.Module):
    def __init__(self,color=False):
        super(CrossEntropy, self).__init__()
        self.color=color

    def forward(self, output, mask):
        #import pdb; pdb.set_trace()
    #Flatten to 1D #Use nn.celoss
        if(self.color):
            #tmp = torch.sigmoid(output)
            if(torch.any(torch.isnan(output)) or torch.any(torch.isnan(mask))):
                import pdb; pdb.set_trace()
            #loss = nn.MSELoss()(output,mask)
            loss = nn.BCELoss(reduction='mean')(output,mask)
            if(torch.any(torch.isnan(loss))):
                import pdb; pdb.set_trace()
            return loss
        else:
            tmp = mask[:,0,:,:]
        return nn.BCELoss(reduction='mean')(output,tmp.view(tmp.shape[0],1,tmp.shape[1],tmp.shape[2]))
        
        
        
        
