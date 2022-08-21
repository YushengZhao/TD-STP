import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (PretrainedConfig, AutoTokenizer)

from utils.misc import length2mask
from reverie.vlnbert_navref import NavRefCMT


def get_tokenizer(args):
    if args.tokenizer == 'bert':
        tokenizer = AutoTokenizer.from_pretrained('/root/mount/Matterport3DSimulator/tdstp/tokenizer_files/bert-base-uncase')
    elif args.tokenizer == 'xlm':
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    else:
        raise NotImplementedError('unsupported tokenizer %s' % args.tokenizer)
    return tokenizer


def get_vlnbert_models(args, config=None):
    model_class = NavRefCMT

    model_name_or_path = args.bert_ckpt_file
    new_ckpt_weights = {}
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path)
        for k, v in ckpt_weights.items():
            if k.startswith('module'):
                new_ckpt_weights[k[7:]] = v
            else:
                # add next_action in weights
                if k.startswith('next_action'):
                    k = 'bert.' + k
                new_ckpt_weights[k] = v

    if args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
        vis_config = PretrainedConfig.from_pretrained(cfg_name)
    else:
        vis_config = PretrainedConfig.from_pretrained('/root/mount/Matterport3DSimulator/tdstp/tokenizer_files/bert-base-uncase')

    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.obj_feat_size = args.obj_feat_size
    vis_config.num_l_layers = args.num_l_layers
    vis_config.num_r_layers = 0
    vis_config.num_h_layers = args.num_h_layers
    vis_config.num_x_layers = args.num_x_layers
    vis_config.hist_enc_pano = args.hist_enc_pano
    vis_config.num_h_pano_layers = args.hist_pano_num_layers

    vis_config.fix_lang_embedding = args.fix_lang_embedding
    vis_config.fix_hist_embedding = args.fix_hist_embedding
    vis_config.fix_obs_embedding = args.fix_obs_embedding

    vis_config.update_lang_bert = not args.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1

    vis_config.no_lang_ca = args.no_lang_ca
    vis_config.max_action_steps = 50

    visual_model = model_class.from_pretrained(
        pretrained_model_name_or_path=None,
        config=vis_config,
        state_dict=new_ckpt_weights)

    return visual_model


class NavRefModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        print('\nInitalizing the VLN-BERT model ...')

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)

        self.hidden_size = args.image_feat_size

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-12)
        )
        self.history_mapper = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        ) if not args.no_hist_mapping else None
        self.target_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1)
        )

    def forward(self, mode, txt_ids=None, txt_embeds=None, txt_masks=None,
                hist_img_feats=None, hist_ang_feats=None,
                hist_pano_img_feats=None, hist_pano_ang_feats=None,
                hist_embeds=None, hist_lens=None, ob_step=None,
                ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None, ob_masks=None, ob_position=None,
                obj_feats=None, obj_angles=None, obj_poses=None, obj_masks=None, obj_position=None,
                global_pos_feat=None, graph_mask=None,
                return_states=False, global_pos=None, position=None, vp_dup=None):

        if mode == 'language':
            encoded_sentence = self.vln_bert(mode, txt_ids=txt_ids, txt_masks=txt_masks)
            return encoded_sentence

        elif mode == 'global_pos':
            if type(txt_embeds) == list:
                return self.position_encoder(global_pos) * txt_embeds[0][:, :1, :]
            else:
                return self.position_encoder(global_pos) * txt_embeds[:, :1, :]

        elif mode == 'history':
            # history inputs per step
            if self.args.hist_envdrop:
                if hist_img_feats is not None:
                    hist_img_feats = self.drop_env(hist_img_feats)
                if hist_pano_img_feats is not None:
                    hist_pano_img_feats = self.drop_env(hist_pano_img_feats)
            if ob_step is not None:
                ob_step_ids = torch.LongTensor([ob_step]).cuda()
            else:
                ob_step_ids = None
            hist_embeds = self.vln_bert(mode, hist_img_feats=hist_img_feats,
                                        hist_ang_feats=hist_ang_feats, ob_step_ids=ob_step_ids,
                                        hist_pano_img_feats=hist_pano_img_feats,
                                        hist_pano_ang_feats=hist_pano_ang_feats)
            if position is not None:
                position_emb = self.position_encoder(position)
                hist_embeds = hist_embeds + position_emb
            return hist_embeds

        elif mode == 'visual':
            # hist_embeds = torch.stack(hist_embeds, 1)
            hist_masks = length2mask(hist_lens, size=hist_embeds.size(1)).logical_not()

            if vp_dup is not None:
                hist_masks[:, 1:].masked_fill_(vp_dup, False)

            ob_img_feats = self.drop_env(ob_img_feats)
            ob_position_feat = self.position_encoder(ob_position)
            obj_feats = self.drop_env(obj_feats)
            obj_position_feat = self.position_encoder(obj_position)

            if global_pos_feat is not None:
                act_logits, obj_logits, txt_embeds, hist_embeds, ob_embeds, obj_embes, pos_embeds = self.vln_bert(
                    mode, txt_embeds=txt_embeds, txt_masks=txt_masks,
                    hist_embeds=hist_embeds, hist_masks=hist_masks,
                    ob_img_feats=ob_img_feats, ob_ang_feats=ob_ang_feats,
                    ob_nav_types=ob_nav_types, ob_masks=ob_masks, ob_position_feat=ob_position_feat,
                    obj_feats=obj_feats, obj_angles=obj_angles,
                    obj_poses=obj_poses, obj_masks=obj_masks, obj_position_feat=obj_position_feat,
                    graph_mask=graph_mask, global_pos_feat=global_pos_feat, history_mapper=self.history_mapper,
                )

                pos_logit = self.target_predictor(pos_embeds * txt_embeds[:, :1]).squeeze(-1)
                # bs x n_pos x hidden -> bs x n_pos

                outs = {'act_logits': act_logits, 'obj_logits': obj_logits,
                        'pos_logits': pos_logit, 'hist_embeds': hist_embeds, 'pos_embeds': pos_embeds}
                if return_states:
                    if self.args.no_lang_ca:
                        states = hist_embeds[:, 0]
                    else:
                        states = txt_embeds[:, 0] * hist_embeds[:, 0]  # [CLS]
                    outs['states'] = states
                return outs
            else:
                act_logits, obj_logits, txt_embeds, hist_embeds, ob_embeds, obj_embeds = self.vln_bert(
                    mode, txt_embeds=txt_embeds, txt_masks=txt_masks,
                    hist_embeds=hist_embeds, hist_masks=hist_masks,
                    ob_img_feats=ob_img_feats, ob_ang_feats=ob_ang_feats,
                    ob_nav_types=ob_nav_types, ob_masks=ob_masks, ob_position_feat=ob_position_feat,
                    obj_feats=obj_feats, obj_angles=obj_angles,
                    obj_poses=obj_poses, obj_masks=obj_masks, obj_position_feat=obj_position_feat,
                    graph_mask=graph_mask, global_pos_feat=global_pos_feat, history_mapper=self.history_mapper,
                )

                outs = {'act_logits': act_logits, 'obj_logits': obj_logits,
                        'hist_embeds': hist_embeds, }
                if return_states:
                    if self.args.no_lang_ca:
                        states = hist_embeds[:, 0]
                    else:
                        states = txt_embeds[:, 0] * hist_embeds[:, 0]  # [CLS]
                    outs['states'] = states
                return outs


class Critic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()
