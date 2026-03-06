import argparse
import torch
import utils
from dataset_sem import FSSiSAIDDataset
from evaluation import Evaluator
from ICPD_Res import ICPD_Net
from logger import AverageMeter

def validate(model, dataloader, device):
    model.eval()
    average_meter = AverageMeter(dataloader.dataset, device)
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = utils.to_cuda(batch, device=device)
            logit_mask, _ = model(batch['query_img'], batch['support_imgs'], batch['support_masks'], batch['sem_ebd'])
            pred_mask = logit_mask.argmax(1)
            
            area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
            
            # 打印进度 (每100个batch)
            if idx % 100 == 0:
                print(f'Validation: [{idx}/{len(dataloader)}]', end='\r')
    
    miou, fb_iou = average_meter.compute_iou()
    print(f'\nValidation Results: mIoU = {miou:.4f}, fB-IoU = {fb_iou:.4f}')
    return miou, fb_iou

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Validation Script')
    parser.add_argument('--model_path', type=str, default='/home/homenew/user1pro/yeh/PCFNet/logs/0.7clip_1s_2f_ISAID/best_model.pt')
    parser.add_argument('--datapath', type=str, default="/home/homenew/user1pro/yeh/Data/iSAID_patches/")
    parser.add_argument('--benchmark', type=str, default='iSAID', choices=['pascal', 'coco', 'fss', 'iSAID', 'DLRSD'])
    parser.add_argument('--bsz', type=int, default=8)
    parser.add_argument('--fold', type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'vgg16'])
    args = parser.parse_args()

    # 初始化数据集
    FSSiSAIDDataset.initialize(datapath=args.datapath)
    dataloader_val = FSSiSAIDDataset.build_dataloader(
        args.benchmark, args.bsz, 2, args.fold, 'val', args.shot
    )
    
    # 初始化模型
    model = ICPD_Net(args.backbone, shot=args.shot, use_original_imgsize=False)
    model.load_state_dict(torch.load(args.model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 初始化评估器
    Evaluator.initialize()
    
    # 执行验证
    validate(model, dataloader_val, device)