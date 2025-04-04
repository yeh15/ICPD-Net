import torch, open_clip
from PIL import Image
import numpy as np

model_name = 'ViT-L-14' # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)

tokenizer = open_clip.get_tokenizer(model_name)

ckpt = torch.load("/home/homenew/user1pro/yeh/PCFNet/checkpoints/RemoteCLIP-ViT-L-14.pt", map_location="cpu")
message = model.load_state_dict(ckpt)
print(message)

model = model.eval()
#model = model.eval()
model_visual = model.visual


# text_queries = [
#     "ship",
#     "storage tank",
#     "baseball diamond",
#     "tennis court",
#     "basketball court",
#     "ground track field",
#     "bridge",
#     "large vehicle",
#     "small vehicle",
#     "helicopter",
#     "swimming pool",
#     "roundabout",
#     "soccer ball field",
#     "plane",
#     "harbor",
#     ]
text_queries = [
    "airplane",
    "bare soil",
    "buildings",
    "cars",
    "chaparral",
    "court",
    "dock",
    "field",
    "grass",
    "mobile home",
    "pavement",
    "sand",
    "sea",
    "ship",
    "storage tanks",
    ]
#formatted_queries = [f"a remote sensing image of {query}" for query in text_queries]
text = tokenizer(text_queries)
image = preprocess(Image.open("/home/homenew/user1pro/yeh/PCFNet/assets/P0010_0_256_412_668.png")).unsqueeze(0)


with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    torch.save(text_features,'dlrsd_vit_L14_sem_embed.pt')
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
    print(1)
print(f'Predictions of {model_name}:')
for query, prob in zip(text_queries, text_probs):
    print(f"{query:<40} {prob * 100:5.1f}%")