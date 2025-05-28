import open_clip
from scipy.ndimage import zoom
import cv2
import torch
from PIL import Image
from GradCam import GradCam
import numpy as np
import os
from tqdm import tqdm
if __name__ == "__main__":
    #cross modal
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14')
    ckpt = torch.load("/home/homenew/user1pro/yeh/PCFNet/checkpoints/RemoteCLIP-ViT-L-14.pt", map_location="cpu")
    model.load_state_dict(ckpt)
    target_layer = model.visual.transformer.resblocks[-1].ln_1
    grad_cam = GradCam(model,target_layer,device='cuda:5')
    text_embeddings = torch.load('/home/homenew/user1pro/yeh/PCFNet/isaid_vit_L14_sem_embed.pt')
    
    train_img_path = '/home/homenew/user1pro/yeh/Data/iSAID_patches/trn/images'
    val_img_path = '/home/homenew/user1pro/yeh/Data/iSAID_patches/val/images'
    cams_path = '/home/homenew/user1pro/yeh/Data/iSAID_patches/cam'
    train_images = os.listdir(train_img_path)
    val_images = os.listdir(val_img_path)
    for train_image in tqdm(train_images):
        image_path = os.path.join(train_img_path,train_image)
        image = preprocess(Image.open(image_path)).unsqueeze(0)
        cams = []
        for class_sample in range(text_embeddings.size(0)):
            selected_row = text_embeddings[class_sample].unsqueeze(0)
            remaining_rows = torch.cat([text_embeddings[:class_sample], text_embeddings[class_sample+1:]], dim=0)
            sem_ebd = torch.cat([selected_row, remaining_rows], dim=0).unsqueeze(0)
            cam = grad_cam(input_img=image,text_embeds=sem_ebd).cpu().numpy()
            cams.append(cam)
        cams = np.stack(cams)
        filename_without_ext, ext = os.path.splitext(train_image)
        save_path = os.path.join(cams_path,filename_without_ext+'.npy')
        np.save(save_path, cams)
        
    for val_image in tqdm(val_images):
        image_path = os.path.join(val_img_path,val_image)
        image = preprocess(Image.open(image_path)).unsqueeze(0)
        cams = []
        for class_sample in range(text_embeddings.size(0)):
            selected_row = text_embeddings[class_sample].unsqueeze(0)
            remaining_rows = torch.cat([text_embeddings[:class_sample], text_embeddings[class_sample+1:]], dim=0)
            sem_ebd = torch.cat([selected_row, remaining_rows], dim=0).unsqueeze(0)
            cam = grad_cam(input_img=image,text_embeds=sem_ebd).cpu().numpy()
            cams.append(cam)
        cams = np.stack(cams)
        filename_without_ext, ext = os.path.splitext(val_image)
        save_path = os.path.join(cams_path,filename_without_ext+'.npy')
        np.save(save_path, cams)
