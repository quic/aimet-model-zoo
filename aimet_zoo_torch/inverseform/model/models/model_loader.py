# pylint: skip-file
# pylint: skip-file
def load_model(net, pretrained):
    pretrained_dict = pretrained['state_dict']
    model_dict = net.state_dict()
    updated_model_dict = {}
    lookup_table = {}
    for k_model, v_model in model_dict.items():
        if k_model.startswith('backbone'):
            k_updated = '.'.join(k_model.split('.')[1:])

            lookup_table[k_updated] = k_model
            updated_model_dict[k_updated] = k_model
        else:
            lookup_table[k_model] = k_model
            updated_model_dict[k_model] = k_model
    
    updated_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('model') or k.startswith('modules') or k.startswith('module'):
            k = '.'.join(k.split('.')[1:])
        if k.startswith('backbone'):
            k = '.'.join(k.split('.')[1:])
       
        if k in updated_model_dict.keys() and model_dict[lookup_table[k]].shape==v.shape:
            updated_pretrained_dict[updated_model_dict[k]] = v
        

    model_dict.update(updated_pretrained_dict)
    net.load_state_dict(model_dict)
    return net