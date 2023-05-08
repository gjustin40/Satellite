import torch

def load_pretrained_weight(model, weight_path):
    checkpoint = torch.load(weight_path, map_location='cpu')
    keys = checkpoint.keys()
    if 'model' in keys:
        pretrained_dict = checkpoint['model']
    elif 'state_dict' in keys:
        pretrained_dict = checkpoint['state_dict']

    model_dict = model.backbone.state_dict()
    new_pretrained_dict = {}

    matched, size_mismatched, not_found = 0, 0, 0
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if model_dict[k].size() == v.size(): # if matched
                new_pretrained_dict[k] = v
                # print(f"Key {k} ------ is matched\n")
                matched += 1
            else:
                # print(f"Size mismatch for {k}. \nModel size: {model_dict[k].size()}, pretrained size: {v.size()}\n")
                size_mismatched += 1
        else:
            # print(f"Key {k} not found in model.")
            not_found += 1

    model_dict.update(new_pretrained_dict)
    model.backbone.load_state_dict(model_dict)

    print('Result of loading Pretrained weight')
    print(f'matched: {matched} | size_mismatched: {size_mismatched} | not_found: {not_found}')
    return model

if __name__ == '__main__':
    from models.beit_adapter_upernet_aux import BEiTAdapterUperNetAux
    checkpoint_path = '/home/yh.sakong/github/distillation/pretrained/beit_large_patch16_224_pt22k_ft22k.pth'
    pretrained_dict = torch.load(checkpoint_path, map_location='cpu')['model']

    model = BEiTAdapterUperNetAux(num_classes=1) # (B, C, H, W)
    model = load_pretrained_weight(model, pretrained_dict)


