import torch
import torch.nn.functional as F
def knn_torch(feature_B, feature_A, mask, k):
    b, ch, nA = feature_A.shape
    feature_A = feature_A.view(b, ch, -1).permute(0, 2, 1).contiguous()  # shape: (b, nA, ch)
    feature_B = feature_B.view(b, ch, -1).permute(0, 2, 1).contiguous()  # shape: (b, nB, ch)

    similarities = []
    for i in range(b):
        dist_squared = torch.cdist(feature_A[i], feature_B[i], p=2).pow(2)  # shape: (nA, nB)
        
        values, indices = torch.topk(-dist_squared, k=k, dim=1, largest=True, sorted=False)
        indices = indices.t().contiguous().flatten()
        
        similarity = mask[i][indices]
        similarity = similarity.view(k, -1).unsqueeze(0)
        similarities.append(similarity)
    similarities = torch.cat(similarities, dim=0)
    return similarities

def min_max_normalize(tensor):

    min_vals = torch.amin(tensor, dim=(1, 2), keepdim=True)
    max_vals = torch.amax(tensor, dim=(1, 2), keepdim=True)

    # 防止除以零的情况
    range_vals = max_vals - min_vals
    mask = range_vals == 0
    range_vals[mask] = 1


    normalized_tensor = (tensor - min_vals) / range_vals
    return normalized_tensor

def semantic_cosine_similarity(query_feat, sem_embedding):

    batch_size, H, W, dim = query_feat.size()
    query_feat_normalized = F.normalize(query_feat, p=2, dim=1)
    sem_embedding_normalized = F.normalize(sem_embedding, p=2, dim=1)
    # query_feat_normalized = query_feat
    # sem_embedding_normalized = sem_embedding
    sem_embedding_expanded = sem_embedding_normalized.unsqueeze(1).unsqueeze(2)  # (4, 1, 1, 15, 768)
    query_feat_expanded = query_feat_normalized.unsqueeze(-2)  # (4, 16, 16, 1, 768)
    cos_sim = F.cosine_similarity(query_feat_expanded, sem_embedding_expanded, dim=-1)  # (4, 16, 16, 15)
    return cos_sim

def cosine_similarity(query_feat, support_feat, mask):
    eps = 1e-5
    support_feat = support_feat * mask
    bsz, ch, hb, wb = support_feat.size()
    support_feat = support_feat.view(bsz, ch, -1)
    support_feat_norm = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

    bsz, ch, ha, wa = query_feat.size()
    query_feat = query_feat.view(bsz, ch, -1)
    query_feat_norm = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

    corr = torch.bmm(query_feat_norm.transpose(1, 2), support_feat_norm)
    corr = corr.clamp(min=0)
    area = torch.sum(mask.view(bsz, -1), dim=1).view(bsz, 1) + eps
    corr = corr.sum(dim=-1)
    corr = corr / area

    corr = corr.view(bsz, ha, wa)
    return corr

def euclid_distance(query_feat, support_feat, mask, k=10):
    bsz, ch, ha, wa = query_feat.size()
    query_feat = query_feat.view(bsz, ch, -1)
    support_feat = support_feat.view(bsz, ch, -1)
    mask = mask.view(bsz, -1)

    with torch.no_grad():
        similarities = knn_torch(support_feat, query_feat, mask, k)
    similarities = similarities.mean(dim=1).view(bsz, ha, wa)
    return similarities


def calculate_dcg(relevance_scores, k):
    ###log###
    gains = (2 ** relevance_scores - 1) / torch.log2(torch.arange(1, k + 1, device=relevance_scores.device).float() + 1)
    return torch.sum(gains, dim=1)
    
def calculate_idcg(k, device):
    ideal_relevance_scores = torch.tensor([1] * k, dtype=torch.float32, device=device)
    return calculate_dcg(ideal_relevance_scores.unsqueeze(0), k)[0]  # 确保形状为 (1,)

def calculate_ndcg(relevance_scores, k):
    if k == 0:
        return torch.zeros((relevance_scores.shape[0],), dtype=torch.float32, device=relevance_scores.device)
    dcg = calculate_dcg(relevance_scores, k)
    idcg = calculate_idcg(k, relevance_scores.device)
    return dcg / idcg

def ndcg_torch(feature_B, feature_A, mask):
    b, ch, nA = feature_A.shape
    nB = feature_B.shape[-1]
    feature_A = feature_A.view(b, ch, -1).permute(0, 2, 1).contiguous()  # shape: (b, nA, ch)
    feature_B = feature_B.view(b, ch, -1).permute(0, 2, 1).contiguous()  # shape: (b, nB, ch)

    ndcgs = []
    for i in range(b):
        dist_squared = torch.cdist(feature_A[i], feature_B[i], p=2).pow(2)  # shape: (nA, nB)
        
        # 动态确定 k
        mask_bool = mask[i] > 0
        k = mask_bool.sum().item()
        
        if k == 0:
            ndcg_values = torch.zeros((nA,), dtype=torch.float32, device=feature_A.device)
        else:
            values, indices = torch.topk(-dist_squared, k=k, dim=1, largest=True, sorted=True)
            relevance_scores = mask_bool[indices].float()  # shape: (nA, k)
            ndcg_values = calculate_ndcg(relevance_scores, k)
        ndcgs.append(ndcg_values.unsqueeze(0))  # shape: (1, nA)
    ndcgs = torch.cat(ndcgs, dim=0)  # shape: (b, nA)
    return ndcgs

def ndcg_corr(query_feat, support_feat, mask):
    bsz, ch, ha, wa = query_feat.size()
    query_feat = query_feat.view(bsz, ch, -1)
    support_feat = support_feat.view(bsz, ch, -1)
    mask = mask.view(bsz, -1)

    with torch.no_grad():
        similarity = ndcg_torch(support_feat, query_feat, mask).view(bsz, ha, wa)
    return similarity

class Correlation:

    @classmethod
    def correlation(cls, query_feats, support_feats, stack_ids, support_mask):
        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                 align_corners=True)
            corr = cosine_similarity(query_feat, support_feat, mask)
            corrs.append(corr)
            #similarity = euclid_distance(query_feat, support_feat, mask)
            similarity = ndcg_corr(query_feat, support_feat, mask)
            corrs.append(similarity)

        corr_l4 = torch.stack(corrs[-stack_ids[0] * 2:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-stack_ids[1] * 2:-stack_ids[0] * 2]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-stack_ids[2] * 2:-stack_ids[1] * 2]).transpose(0, 1).contiguous()

        return [corr_l4, corr_l3, corr_l2]
    @classmethod
    def sem_correlation(cls, query_feats, support_feats, stack_ids, support_mask, sem_ebeddings):
        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                 align_corners=True)
            corr = cosine_similarity(query_feat, support_feat, mask)
            corrs.append(corr)
            #similarity = euclid_distance(query_feat, support_feat, mask)
            similarity = ndcg_corr(query_feat, support_feat, mask)
            corrs.append(similarity)
        sem_sim = semantic_cosine_similarity(query_feat,sem_ebeddings)
        corrs.append(sem_sim)
        corr_l4 = torch.stack(corrs[-2*stack_ids[0] -1:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-2*stack_ids[1] -1:-2*stack_ids[0] -1]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-2*stack_ids[2] -1:-2*stack_ids[1] -1]).transpose(0, 1).contiguous()
        # corr_l4 = torch.stack(corrs[-2*stack_ids[0] :]).transpose(0, 1).contiguous()
        # corr_l3 = torch.stack(corrs[-2*stack_ids[1] :-2*stack_ids[0] ]).transpose(0, 1).contiguous()
        # corr_l2 = torch.stack(corrs[-2*stack_ids[2] :-2*stack_ids[1] ]).transpose(0, 1).contiguous()
        return [corr_l4, corr_l3, corr_l2]
    
    @classmethod
    def clip_correlation(cls, query_CM_feats, spt_CM_feats, support_mask):
        corrs = []
        query_feat, support_feat = query_CM_feats.permute(0, 3, 1, 2), spt_CM_feats.permute(0, 3, 1, 2)
        mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                align_corners=True)
        corr = cosine_similarity(query_feat, support_feat, mask)
        corrs.append(corr)
        similarity = ndcg_corr(query_feat, support_feat, mask)
        corrs.append(similarity)
        corrs = torch.stack(corrs).transpose(0, 1).contiguous()
        return corrs
