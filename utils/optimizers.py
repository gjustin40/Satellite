def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed', 'backbone.visual_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed') or \
            var_name.startswith('backbone.visual_embed'):
        return 0
    elif var_name.startswith('decode_head.mask_embed'):
        return 0
    elif var_name.startswith('decode_head.cls_embed'):
        return 0
    elif var_name.startswith('decode_head.level_embed'):
        return 0
    elif var_name.startswith('decode_head.query_embed'):
        return 0
    elif var_name.startswith('decode_head.query_feat'):
        return 0
    elif var_name.startswith('backbone.blocks') or \
            var_name.startswith('backbone.layers'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1

    else:
        return num_max_layer - 1

def layer_decay_optimizer_constructor(opt, model):
    base_lr = opt.OPTIM.LR
    weight_decay = opt.OPTIM.WEIGHT_DECAY
    paramwise_cfg = opt.OPTIM.PARAMWISE

    parameter_groups = {}
    num_layers = paramwise_cfg.get('NUM_LAYERS') + 2
    layer_decay_rate = paramwise_cfg.get('LAYER_DECAY_RATE')

    if hasattr(model, 'module'):
        model = model.module
    
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') \
                or name in ('pos_embed', 'cls_token', 'visual_embed'):
            # or "relative_position_bias_table" in name:
            group_name = 'no_decay'
            this_weight_decay = 0.
        else:
            group_name = 'decay'
            this_weight_decay = weight_decay

        layer_id = get_num_layer_for_vit(name, num_layers)
        group_name = 'layer_%d_%s' % (layer_id, group_name)

        if group_name not in parameter_groups:
            scale = layer_decay_rate**(num_layers - layer_id - 1)

            parameter_groups[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'param_names': [],
                'lr_scale': scale,
                'group_name': group_name,
                'lr': scale * base_lr,
            }

        parameter_groups[group_name]['params'].append(param)
        parameter_groups[group_name]['param_names'].append(name)

    params.extend(parameter_groups.values())
    return params
