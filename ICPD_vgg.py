import torch
from torch import nn
import torch.nn.functional as F
from HSNet import extract_feat_res, extract_feat_vgg, extract_feat_modified_res,cross_modal_feats_vit_l14
from correlation import Correlation,semantic_cosine_similarity
from torchvision.models import resnet
from torchvision.models import vgg
from functools import reduce
from operator import add
import open_clip
import numpy as np
import torchvision.transforms as transforms
from CAM.GradCam import GradCam

class HPNLearner(nn.Module):
    def __init__(self, inch):
        super(HPNLearner, self).__init__()

        def make_building_block(in_channel, out_channels, group=4):

            building_block_layers = []
            for idx, outch in enumerate(out_channels):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                building_block_layers.append(nn.Conv2d(inch, outch, 3, 1, 1))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3 = 16, 64, 128

        # Squeezing building blocks
        #self.encoder_layer4 = make_building_block(2*(inch[0]+1)+1, [outch1, outch2, outch3])
        self.encoder_layer4 = make_building_block(2*inch[0], [outch1, outch2, outch3])
        self.encoder_layer3 = make_building_block(2*inch[1]+3, [outch1, outch2, outch3])
        self.encoder_layer2 = make_building_block(2*inch[2], [outch1, outch2, outch3])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3])
        self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch3])

        # Decoder layers
        self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))
    
    def forward(self, hypercorr_pyramid):

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])

        # Propagate encoded 4D-tensor (Mixing building blocks)
        hypercorr_sqz4 = F.interpolate(hypercorr_sqz4, hypercorr_sqz3.shape[-2:], mode='bilinear', align_corners=True)
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        hypercorr_mix43 = F.interpolate(hypercorr_mix43, hypercorr_sqz2.shape[-2:], mode='bilinear', align_corners=True)
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)


        # Decode the encoded 4D-tensor
        hypercorr_decoded = self.decoder1(hypercorr_mix432)
        upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.decoder2(hypercorr_decoded)

        return logit_mask
class PriorLearner(nn.Module):
    def __init__(self, in_channels=15, out_channels=1):
        super(PriorLearner, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 32, kernel_size=2, stride=2),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.InstanceNorm2d(16),
            nn.ReLU()
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        upsampled = self.upsample(x)
        downsampled = self.downsample(upsampled)
        output = self.sigmoid(downsampled)
        output = output.permute(0, 2, 3, 1)
        return output

class IDPD_Net(nn.Module):
    def __init__(self, backbone, shot, use_original_imgsize):
        super(IDPD_Net, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'vgg16':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14')
            ckpt = torch.load("/home/homenew/user1pro/yeh/PCFNet/checkpoints/RemoteCLIP-ViT-L-14.pt", map_location="cpu")
            model.load_state_dict(ckpt)
            self.cross_modal_visual_backbone = model.visual
            self.cross_modal_feats = cross_modal_feats_vit_l14
            
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            #cross modal
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14')
            ckpt = torch.load("/home/homenew/user1pro/yeh/PCFNet/checkpoints/RemoteCLIP-ViT-L-14.pt", map_location="cpu")
            model.load_state_dict(ckpt)
            #target_layer = model.visual.transformer.resblocks[-1].ln_1
            #self.gradcam = GradCam(model,target_layer)
            self.cross_modal_visual_backbone = model.visual
            #pure visual branch
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            
            self.cross_modal_feats = cross_modal_feats_vit_l14
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]

        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        #self.stack_ids = [4,10,14]
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.priorlearner = PriorLearner(in_channels=15,out_channels=1)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        
    @staticmethod
    def prior_criterion(spt_binary_priors_list, support_mask):
        batch_size = support_mask.size(0)
        shots = len(spt_binary_priors_list)
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        total_loss = 0
        for i in range(shots):
            prior_upsampled = F.interpolate(
                spt_binary_priors_list[i].permute(0, 3, 1, 2),
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            current_mask = support_mask[:, i]

            loss = criterion(prior_upsampled, current_mask.float())
            weighted_loss = loss
            total_loss += weighted_loss.mean()
        return total_loss / shots

    @staticmethod
    def weighted_sum_with_mask(sptfeat, spt_mask, fusion_query_proto):
        """
        根据 hiermask 筛选 sptfeat 中的特征，并与 fusion_query_proto 进行加权求和。
        
        参数:
        - sptfeat: 形状为 (bsz, dim, h, w) 的张量。
        - hiermask: 形状为 (bsz, 1, h, w) 的掩码张量。
        - fusion_query_proto: 形状为 (bsz, dim) 的原型张量。
        
        返回:
        - 加权求和后的张量，形状为 (bsz, dim, h, w)
        """
        if (fusion_query_proto == 0).all():
            return sptfeat
        bsz, dim, h, w = sptfeat.shape
        fusion_query_proto_expanded = fusion_query_proto.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        spt_mask = F.interpolate(spt_mask.float(), size=(h, w), mode='bilinear', align_corners=False)
        mask = spt_mask > 0
        #weighted_sum = 0.5 * sptfeat + 0.5 * fusion_query_proto_expanded
        weighted_sum = sptfeat
        result = torch.where(mask.expand_as(weighted_sum), weighted_sum, sptfeat)
        return result
    @staticmethod
    def min_max_normalize(tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input should be a torch.Tensor.")
        
        shape = tensor.shape
        batch_size, H, W = shape
        reshaped_tensor = tensor.view(batch_size, -1)
        min_vals, _ = reshaped_tensor.min(dim=1, keepdim=True)
        max_vals, _ = reshaped_tensor.max(dim=1, keepdim=True)
        
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        

        normalized_tensor = (reshaped_tensor - min_vals) / range_vals
        normalized_tensor = normalized_tensor.view(shape)
        return normalized_tensor
    def forward(self, query_img, support_img, support_mask, semantic_embeddings):
        with torch.no_grad():
            resize_transform = transforms.Resize(size=(224, 224), antialias=True)
            support_feats_list = []
            spt_CM_feats_list = []
            #backbone hierarchy feats
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            #query cross modal feats
            query_CM_feats = self.cross_modal_feats(resize_transform(query_img),self.cross_modal_visual_backbone)
            for i in range(self.shot):
                support_feats = self.extract_feats(support_img[:, i, :, :, :], self.backbone, self.feat_ids,
                                               self.bottleneck_ids, self.lids)
                spt_CM_feats = self.cross_modal_feats(resize_transform(support_img[:, i, :, :, :]), self.cross_modal_visual_backbone)
                support_feats_list.append(support_feats)
                spt_CM_feats_list.append(spt_CM_feats)
        query_multi_cls_priors = semantic_cosine_similarity(query_CM_feats,semantic_embeddings)
        query_clip_priors = self.priorlearner(query_multi_cls_priors)
        spt_binary_priors_list = []
        for i in range(self.shot):
            spt_multi_cls_priors = semantic_cosine_similarity(spt_CM_feats_list[i],semantic_embeddings)
            spt_binary_priors = self.priorlearner(spt_multi_cls_priors)
            spt_binary_priors_list.append(spt_binary_priors)
        prior_loss = self.prior_criterion(spt_binary_priors_list,support_mask)
        ###query2support feature fusion###
        hierarchy_size = [32,16,8]
        query_priors = query_clip_priors.squeeze(-1)
        hier_query_priors = []
        query_priors_unsq = query_priors.unsqueeze(1)
        for size in hierarchy_size:
            query_priors_resized = F.interpolate(query_priors_unsq, size=(size, size), mode='bilinear', align_corners=False).squeeze(1)
            query_priors_resized = self.min_max_normalize(query_priors_resized)
            #hier_query_prior = torch.nn.functional.softmax(query_priors_resized.view(query_priors_resized.shape[0], -1), dim=1).view_as(query_priors_resized)
            hier_query_priors.append(query_priors_resized)

        levels = [0,0,0,1,1,1,2]
        fusion_query_protos = []
        for i in range(len(query_feats)):
            prior_mask = hier_query_priors[levels[i]].unsqueeze(1)
            queryfeat = query_feats[i]
            mask = prior_mask > 0.7
            if mask.any():
                masked_weighted_query_feat = queryfeat * prior_mask * mask.float()
                sum_masked_features = masked_weighted_query_feat.sum(dim=(2, 3))
                num_elements = mask.sum(dim=(2, 3)).float()
                num_elements[num_elements == 0] = 1
                fusion_query_proto = sum_masked_features / num_elements
            else:
                shape = list(queryfeat.shape)
                shape[2] = shape[3] = 1
                fusion_query_proto = torch.zeros(shape, device=queryfeat.device)
            fusion_query_protos.append(fusion_query_proto)
        
        prior_mask = query_priors.unsqueeze(1)
        queryfeat = query_CM_feats.permute(0, 3, 1, 2)
        mask = prior_mask > 0.7
        if mask.any():
            masked_weighted_query_feat = queryfeat * prior_mask * mask.float()
            sum_masked_features = masked_weighted_query_feat.sum(dim=(2, 3))
            num_elements = mask.sum(dim=(2, 3)).float()
            num_elements[num_elements == 0] = 1
            fusion_query_proto = sum_masked_features / num_elements
        else:
            shape = list(queryfeat.shape)
            shape[2] = shape[3] = 1
            fusion_query_proto = torch.zeros(shape, device=queryfeat.device)
        fusion_query_CM_proto = fusion_query_proto
        
        for k in range(self.shot):
            spt_CM_feats_list[k]=self.weighted_sum_with_mask(spt_CM_feats_list[k].permute(0, 3, 1, 2),support_mask[:, k, :, :].clone().unsqueeze(1),fusion_query_CM_proto).permute(0,2,3,1)
            for i in range(len(query_feats)):
                support_feats_list[k][i]=self.weighted_sum_with_mask(support_feats_list[k][i],support_mask[:, k, :, :].clone().unsqueeze(1),fusion_query_protos[i])       
        
        
        corr_list = []
        for i in range(self.shot):
            corr = Correlation.correlation(query_feats, support_feats_list[i], self.stack_ids,
                                           support_mask[:, i, :, :].clone())
            clip_corr = Correlation.clip_correlation(query_CM_feats, spt_CM_feats_list[i], support_mask[:, i, :, :].clone())
            corr[1] = torch.cat((corr[1], clip_corr,query_priors_unsq), dim=1)
            corr_list.append(corr)
        corr = corr_list[0]
        if self.shot > 1:
            for i in range(1, self.shot):
                corr[0] += corr_list[i][0]
                corr[1] += corr_list[i][1]
                corr[2] += corr_list[i][2]
            corr[0] /= self.shot
            corr[1] /= self.shot
            corr[2] /= self.shot
        logit_mask = self.hpn_learner(corr)
        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask, support_img.size()[-2:], mode='bilinear', align_corners=True)
        return logit_mask,prior_loss


    def compute_objective(self, logit_mask, gt_mask, scale_score):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        scale_score = scale_score.view(bsz, -1)
        gt_mask = gt_mask.view(bsz, -1).long()
        loss_pixel = self.cross_entropy_loss(logit_mask, gt_mask)
        loss_pixel = (loss_pixel * scale_score).mean()
        return loss_pixel

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging


if __name__ == "__main__":
    batch = {
        'query_img': torch.rand(4, 3, 256, 256),
        'support_imgs': torch.rand(4, 5, 3, 256, 256),
        'support_masks': torch.randint(0, 2, (4, 5, 256, 256)),
        'org_query_imsize': (256, 256),
        'semantic_embeddings': torch.rand(4,15,768)
    }
    model = IDPD_Net('vgg16', 5, False)
    out,_ = model(batch['query_img'], batch['support_imgs'], batch['support_masks'], batch['semantic_embeddings'])
    print(out.size())