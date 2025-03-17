from causal_earth.models import mae_vit_large_patch16_dec512d8b
import torch
from causal_earth.cfgs.train import MAEConfig
from causal_earth.utils.model_utils import interpolate_pos_embed
import draccus


def model_check(model1, model2):
    nodes = [(model1, model2)]

    while len(nodes) > 0:
        model1, model2 = nodes.pop()
        if type(model1) != dict and type(model2) != dict:
            continue
        elif type(model1) != dict or type(model2) != dict:
            print('uneven depth')


@draccus.wrap()
def main(cfg: MAEConfig):

    model = mae_vit_large_patch16_dec512d8b()
    checkpoint = torch.load(cfg.ckpt_path, map_location='cpu')
    
    print("Load pre-trained checkpoint from: %s" % cfg.ckpt_path)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()

    
    # if 'patch_embed.proj.weight' in checkpoint_model and 'patch_embed.proj.weight' in state_dict:
    #     ckpt_patch_embed_weight = checkpoint_model['patch_embed.proj.weight']
    #     model_patch_embed_weight = state_dict['patch_embed.proj.weight']
    #     if ckpt_patch_embed_weight.shape[1] != model_patch_embed_weight.shape[1]:
    #         print('Using 3 channels of ckpt patch_embed')
    #         model.patch_embed.proj.weight.data[:, :3, :, :] = ckpt_patch_embed_weight.data[:, :3, :, :]

    # TODO: Do something smarter?
    for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    breakpoint()


if __name__ == "__main__":
    main()