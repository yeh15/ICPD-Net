import numpy as np
import torch
from typing import Callable, List, Tuple
from PIL import Image
def visual_text_classifier(visual_embedding,text_embeddings):
    visual_embedding = visual_embedding / visual_embedding.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings/text_embeddings.norm(dim=-1, keepdim=True)
    logits = torch.einsum('bd,bcd->bc', visual_embedding, text_embeddings)
    logits = (100*logits).softmax(dim=-1)
    return logits
class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activations = []
        self.handles = []
        self.handles.append(
            target_layer.register_forward_hook(self.save_activation))
        self.handles.append(
            target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):#前向传播到目标层时执行self.save_gradient
        activation = output[:,1:,:]
        result = activation.reshape(activation.size(0),16,16,activation.size(2))
        self.activations.append(result.detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return
        def _store_grad(grad):
            result = grad[:,1:,:]
            result = result.reshape(grad.size(0),16,16,grad.size(2))
            self.gradients =  self.gradients + [result.detach()]
        output.register_hook(_store_grad)

    def __call__(self, image):
        self.gradients = []
        self.activations = []
        return self.model.encode_image(image)
    
    def release(self):
        for handle in self.handles:
            handle.remove()

class GradCam:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layer: torch.nn.Module,
                 device: str = 'cpu',
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model = model.eval()#使用的模型
        self.target_layer = target_layer#目标层
        self.device = device
        if self.device != 'cpu':
            self.model = model.cuda(self.device)
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layer)
    def get_cam_weights(self,
                        grads):
        return grads.mean(dim=(1, 2))
    def get_cam_image(self,
                      activations: torch.Tensor,
                      grads: torch.Tensor):
        weights = self.get_cam_weights(grads).unsqueeze(1).unsqueeze(1)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=-1).squeeze()
        cam = torch.clamp(cam,min=0)
        epsilon = 1e-8
        cam_min = cam.min()
        cam_max = cam.max()

        range = cam_max - cam_min
        cam = (cam - cam_min) / (range + epsilon)
        
        return cam
    def forward(self,
                 input_img: torch.Tensor,
                 text_embeds: List[torch.nn.Module] = None,
                 ) -> np.ndarray:

        if self.device != 'cpu':
            input_img = input_img.cuda(self.device)
            text_embeds= text_embeds.cuda(self.device)
        if self.compute_input_gradient:
            input_img = torch.autograd.Variable(input_img,
                                                   requires_grad=True)
            

        visual_embedding = self.activations_and_grads(input_img)#visual_embedding
        logits = visual_text_classifier(visual_embedding = visual_embedding,text_embeddings=text_embeds)
        if self.uses_gradients:
            self.model.zero_grad()
            loss = logits[0,8]
            loss.backward(retain_graph=True)
        grad_cam = self.get_cam_image(self.activations_and_grads.activations[0],self.activations_and_grads.gradients[0])
        return grad_cam

    def __call__(self,
                 input_img: torch.Tensor,
                 text_embeds: List[torch.nn.Module] = None,
                 ) -> np.ndarray:
        return self.forward(input_img,
                            text_embeds)

import open_clip
from scipy.ndimage import zoom
import cv2
if __name__ == "__main__":
    #cross modal
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14')
    ckpt = torch.load("/home/homenew/user1pro/yeh/PCFNet/checkpoints/RemoteCLIP-ViT-L-14.pt", map_location="cpu")
    model.load_state_dict(ckpt)
    target_layer = model.visual.transformer.resblocks[-1].ln_1
    grad_cam = GradCam(model,target_layer,device='cuda:5')
    
    image_path = "/home/homenew/user1pro/yeh/Data/iSAID_patches/trn/images/P0023_206_462_1030_1286.png"
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    text_embeddings = torch.load('/home/homenew/user1pro/yeh/PCFNet/test_embds.pt').unsqueeze(0)
    cam = grad_cam(input_img=image,text_embeds=text_embeddings)
    
    image = cv2.imread(image_path)  # Read image in BGR format
    # Convert to RGB if necessary (OpenCV loads images in BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / 255.0
    cam = cam.cpu().numpy()
    heatmap_resized = zoom(cam, 256/16, order=1)
    
    def show_cam_on_image(img: np.ndarray,
                        mask: np.ndarray,
                        use_rgb: bool = False,
                        colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """ This function overlays the cam mask on the image as an heatmap.
        By default the heatmap is in BGR format.
        :param img: The base image in RGB or BGR format.
        :param mask: The cam mask.
        :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
        :param colormap: The OpenCV colormap to be used.
        :returns: The default image with the cam overlay.
        """
        
        # Ensure the mask is between 0 and 1
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        
        if np.max(img) > 1:
            raise Exception("The input image should np.float32 in the range [0, 1]")
        
        fusion_img = heatmap + img
        fusion_img = fusion_img / np.max(fusion_img)
        return np.uint8(255 * fusion_img)

    # Now call the function with the processed image and heatmap
    result = show_cam_on_image(image_rgb, heatmap_resized, use_rgb=True)
    # If you want to save or display the result
    cv2.imwrite('heatmap_cam100.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))  # Save the output

