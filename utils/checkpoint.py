import torch

def load_pretrained_weight(model, pretrained_dict):
    model_dict = model.backbone.state_dict()
    new_pretrained_dict = {}

    for k, v in pretrained_dict.items():
        if k in model_dict:
            if model_dict[k].size() == v.size():
                new_pretrained_dict[k] = v
                print(f"Key {k} ------ is matched\n")
            else:
                pass
                print(f"Size mismatch for {k}. \nModel size: {model_dict[k].size()}, pretrained size: {v.size()}\n")
        else:
            pass
            print(f"Key {k} not found in model.")

    model_dict.update(new_pretrained_dict)
    model.backbone.load_state_dict(model_dict)

    return model

if __name__ == '__main__':
    from models.beit_adapter_upernet_aux import BEiTAdapterUperNetAux
    checkpoint_path = '/home/yh.sakong/github/distillation/pretrained/beit_large_patch16_224_pt22k_ft22k.pth'
    pretrained_dict = torch.load(checkpoint_path, map_location='cpu')['model']

    model = BEiTAdapterUperNetAux(num_classes=1) # (B, C, H, W)
    model = load_pretrained_weight(model, pretrained_dict)


