import json
import os
import sys

import networkx as nx
import numpy as np
import random
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.distributed import is_default_gpu
from utils.misc import length2mask
from utils.logger import print_progress

from models.model_TDSTP import VLNBertCMT, Critic

from .eval_utils import cal_dtw

from .agent_base import BaseAgent


class Seq2SeqCMTAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
        'left': (0, -1, 0),  # left
        'right': (0, 1, 0),  # right
        'up': (0, 0, 1),  # up
        'down': (0, 0, -1),  # down
        'forward': (1, 0, 0),  # forward
        '<end>': (0, 0, 0),  # <end>
        '<start>': (0, 0, 0),  # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, rank=0):
        super().__init__(env)
        self.args = args

        self.default_gpu = is_default_gpu(self.args)
        self.rank = rank

        # Models
        self._build_model()

        if self.args.world_size > 1:
            self.vln_bert = DDP(self.vln_bert, device_ids=[self.rank], find_unused_parameters=True)
            self.critic = DDP(self.critic, device_ids=[self.rank], find_unused_parameters=True)

        self.models = (self.vln_bert, self.critic)
        self.device = torch.device('cuda:%d' % self.rank)  # TODO

        # Optimizers
        if self.args.optim == 'rms':
            optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            optimizer = torch.optim.SGD
        else:
            assert False
        if self.default_gpu:
            print('Optimizer: %s' % self.args.optim)

        self.vln_bert_optimizer = optimizer(self.vln_bert.parameters(), lr=self.args.lr)
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr)
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, size_average=False)

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _build_model(self):
        self.vln_bert = VLNBertCMT(self.args).cuda()
        self.critic = Critic(self.args).cuda()

    def _language_variable(self, obs):
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]

        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor)
        mask = torch.from_numpy(mask)
        return seq_tensor.long().cuda(), mask.cuda(), seq_lengths

    def _cand_pano_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        ob_cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        ob_lens = []
        ob_img_fts, ob_ang_fts, ob_nav_types, ob_pos = [], [], [], []
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            cand_img_fts, cand_ang_fts, cand_nav_types, cand_pos = [], [], [], []
            cand_pointids = np.zeros((self.args.views,), dtype=np.bool)
            for j, cc in enumerate(ob['candidate']):
                cand_img_fts.append(cc['feature'][:self.args.image_feat_size])
                cand_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                cand_pointids[cc['pointId']] = True
                cand_nav_types.append(1)
                if self.args.cand_use_ob_pos:
                    cand_pos.append(ob['position'])
                else:
                    cand_pos.append(cc['position'])
            # add [STOP] feature
            cand_img_fts.append(np.zeros((self.args.image_feat_size,), dtype=np.float32))
            cand_ang_fts.append(np.zeros((self.args.angle_feat_size,), dtype=np.float32))
            cand_pos.append(ob['position'])
            cand_img_fts = np.vstack(cand_img_fts)
            cand_ang_fts = np.vstack(cand_ang_fts)
            cand_nav_types.append(2)

            # add pano context
            pano_fts = ob['feature'][~cand_pointids]
            cand_pano_img_fts = np.concatenate([cand_img_fts, pano_fts[:, :self.args.image_feat_size]], 0)
            cand_pano_ang_fts = np.concatenate([cand_ang_fts, pano_fts[:, self.args.image_feat_size:]], 0)
            cand_nav_types.extend([0] * (self.args.views - np.sum(cand_pointids)))
            cand_pos.extend([ob['position'] for _ in range(self.args.views - np.sum(cand_pointids))])

            ob_lens.append(len(cand_nav_types))
            ob_img_fts.append(cand_pano_img_fts)
            ob_ang_fts.append(cand_pano_ang_fts)
            ob_nav_types.append(cand_nav_types)
            ob_pos.append(cand_pos)

        # pad features to max_len
        max_len = max(ob_lens)
        for i in range(len(obs)):
            num_pads = max_len - ob_lens[i]
            ob_img_fts[i] = np.concatenate([ob_img_fts[i], \
                                            np.zeros((num_pads, ob_img_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_ang_fts[i] = np.concatenate([ob_ang_fts[i], \
                                            np.zeros((num_pads, ob_ang_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_nav_types[i] = np.array(ob_nav_types[i] + [0] * num_pads)
            ob_pos[i] = np.array(ob_pos[i] + [np.array([0, 0, 0], dtype=np.float32) for _ in range(num_pads)])

        ob_img_fts = torch.from_numpy(np.stack(ob_img_fts, 0)).cuda()
        ob_ang_fts = torch.from_numpy(np.stack(ob_ang_fts, 0)).cuda()
        ob_nav_types = torch.from_numpy(np.stack(ob_nav_types, 0)).cuda()
        ob_pos = torch.from_numpy(np.stack(ob_pos, 0)).float().cuda()

        return ob_img_fts, ob_ang_fts, ob_nav_types, ob_lens, ob_cand_lens, ob_pos

    def _candidate_variable(self, obs):
        cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        max_len = max(cand_lens)
        cand_img_feats = np.zeros((len(obs), max_len, self.args.image_feat_size), dtype=np.float32)
        cand_ang_feats = np.zeros((len(obs), max_len, self.args.angle_feat_size), dtype=np.float32)
        cand_nav_types = np.zeros((len(obs), max_len), dtype=np.int64)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                cand_img_feats[i, j] = cc['feature'][:self.args.image_feat_size]
                cand_ang_feats[i, j] = cc['feature'][self.args.image_feat_size:]
                cand_nav_types[i, j] = 1
            cand_nav_types[i, cand_lens[i] - 1] = 2

        cand_img_feats = torch.from_numpy(cand_img_feats).cuda()
        cand_ang_feats = torch.from_numpy(cand_ang_feats).cuda()
        cand_nav_types = torch.from_numpy(cand_nav_types).cuda()
        return cand_img_feats, cand_ang_feats, cand_nav_types, cand_lens

    def _history_variable(self, obs):
        hist_img_feats = np.zeros((len(obs), self.args.image_feat_size), np.float32)
        for i, ob in enumerate(obs):
            hist_img_feats[i] = ob['feature'][ob['viewIndex'], :self.args.image_feat_size]
        hist_img_feats = torch.from_numpy(hist_img_feats).cuda()

        if self.args.hist_enc_pano:
            hist_pano_img_feats = np.zeros((len(obs), self.args.views, self.args.image_feat_size), np.float32)
            hist_pano_ang_feats = np.zeros((len(obs), self.args.views, self.args.angle_feat_size), np.float32)
            for i, ob in enumerate(obs):
                hist_pano_img_feats[i] = ob['feature'][:, :self.args.image_feat_size]
                hist_pano_ang_feats[i] = ob['feature'][:, self.args.image_feat_size:]
            hist_pano_img_feats = torch.from_numpy(hist_pano_img_feats).cuda()
            hist_pano_ang_feats = torch.from_numpy(hist_pano_ang_feats).cuda()
        else:
            hist_pano_img_feats, hist_pano_ang_feats = None, None

        return hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats

    def _teacher_action(self, obs, ended, max_hist_len):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:  # Next view point
                        a[i] = k + max_hist_len
                        break
                else:  # Stop here
                    assert ob['teacher'] == ob['viewpoint']  # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate']) + max_hist_len
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, obs, max_hist_len, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """

        def take_action(i, name):
            if type(name) is int:  # Go to the next view
                self.env.env.sims[i].makeAction([name], [0], [0])
            else:  # Adjust
                self.env.env.sims[i].makeAction(*self.env_actions[name])

        def adjust(i, src_point, trg_point):
            src_level = src_point // 12  # The point idx started from 0
            trg_level = trg_point // 12
            while src_level < trg_level:  # Tune up
                take_action(i, 'up')
                src_level += 1
            while src_level > trg_level:  # Tune down
                take_action(i, 'down')
                src_level -= 1
            while self.env.env.sims[i].getState()[0].viewIndex != trg_point:  # Turn right until the target
                take_action(i, 'right')

        for i, ob in enumerate(obs):
            action = a_t[i]
            if action != -1:  # -1 is the <stop> action
                if action >= max_hist_len:
                    action = action - max_hist_len
                    select_candidate = ob['candidate'][action]
                    src_point = ob['viewIndex']
                    trg_point = select_candidate['pointId']
                    adjust(i, src_point, trg_point)
                    state = self.env.env.sims[i].getState()[0]
                    for idx, loc in enumerate(state.navigableLocations):
                        if loc.viewpointId == select_candidate['viewpointId']:
                            take_action(i, idx)
                            state = self.env.env.sims[i].getState()[0]
                            if traj is not None:
                                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
                else:
                    action = action - 1  # 1 for global history token
                    target_vp = self.seq_vp_list[i][action]
                    current_vp = ob['viewpoint']
                    src_point = ob['viewIndex']
                    trg_point = self.seq_view_idx_list[i][action]
                    path = nx.single_source_shortest_path(self.graphs[i], current_vp)[target_vp]
                    state = self.env.env.sims[i].getState()[0]
                    for j in range(len(path) - 1):
                        # from path[j] to path[j+1]
                        for idx, loc in enumerate(state.navigableLocations):
                            if loc.viewpointId == path[j+1]:
                                take_action(i, idx)
                                state = self.env.env.sims[i].getState()[0]
                                if traj is not None:
                                    traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
                                break
                        else:
                            raise ValueError('no navigable location')
                    adjust(i, src_point, trg_point)

    def _init_graph(self, batch_size):
        self.graphs = [nx.Graph() for _ in range(batch_size)]
        self.vp2idx_list = [dict() for _ in range(batch_size)]
        self.seq_idx_list = [list() for _ in range(batch_size)]
        self.seq_vp_list = [list() for _ in range(batch_size)]
        self.seq_view_idx_list = [list() for _ in range(batch_size)]
        self.seq_dist_list = [list() for _ in range(batch_size)]
        self.seq_dup_vp = [list() for _ in range(batch_size)]
        self.seq_last_idx = [dict() for _ in range(batch_size)]
        self.blocked_path = [defaultdict(lambda: defaultdict(lambda: 0)) for _ in range(batch_size)]

    def _update_graph(self, obs, ended, a_t, max_hist_len):
        for i, ob in enumerate(obs):
            if ended[i]:
                self.seq_dup_vp[i].append(True)
                continue
            vp = ob['viewpoint']
            if vp not in self.vp2idx_list[i]:
                idx = len(self.vp2idx_list[i])
                self.vp2idx_list[i][vp] = idx
                self.graphs[i].add_node(vp)
                self.seq_dup_vp[i].append(False)
                self.seq_last_idx[i][vp] = len(self.seq_dup_vp[i]) - 1
            else:
                idx = self.vp2idx_list[i][vp]
                if self.args.no_temporal_strategy == 'replace':
                    self.seq_dup_vp[i].append(False)
                    self.seq_dup_vp[i][self.seq_last_idx[i][vp]] = True
                else:  # 'keep' strategy, keep the old one
                    self.seq_dup_vp[i].append(True)
                self.seq_last_idx[i][vp] = len(self.seq_dup_vp[i]) - 1
            self.seq_idx_list[i].append(idx)
            self.seq_vp_list[i].append(vp)
            self.seq_view_idx_list[i].append(ob['viewIndex'])
            self.seq_dist_list[i].append(ob['distance'])
            for adj in ob['navigableLocations']:
                adj_vp = adj.viewpointId
                if adj_vp in self.vp2idx_list[i]:
                    self.graphs[i].add_edge(vp, adj_vp)

            # block path if backtrack
            if max_hist_len > a_t[i] >= 0:
                hist_vp = self.seq_vp_list[i][a_t[i] - 1]
                self.blocked_path[i][hist_vp][vp] += 1

    def _get_connectivity_mask(self):
        batch_size = len(self.graphs)
        max_size = max([len(seq) for seq in self.seq_idx_list])
        mask = torch.ones((batch_size, max_size, max_size)).cuda()
        for i in range(batch_size):
            adj_matrix = nx.adj_matrix(self.graphs[i], weight=1)
            adj_matrix.setdiag(1)
            adj_matrix = adj_matrix.toarray()
            expanded_matrix = adj_matrix[np.ix_(self.seq_idx_list[i], self.seq_idx_list[i])]
            expanded_matrix = torch.from_numpy(expanded_matrix).cuda()
            node_size = len(self.seq_idx_list[i])
            mask[i, :node_size, :node_size] *= expanded_matrix
        return mask

    def _get_dup_logit_mask(self, obs):
        batch_size = len(self.graphs)
        max_size = max([len(seq) for seq in self.seq_idx_list])
        mask = torch.ones((batch_size, max_size)).cuda()
        for i in range(batch_size):
            for j, vp in enumerate(self.seq_vp_list[i]):
                if vp == obs[i]['viewpoint']:
                    mask[i, j] = 0
                if self.args.no_temporal and self.seq_dup_vp[i][j]:
                    mask[i, j] = 0
        return mask

    def _get_rl_optimal_action(self, obs, ended, max_hist_len, rl_forbidden):
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:
                a[i] = self.args.ignoreid
            else:
                distances = [1e7] * rl_forbidden.size(1)
                if self.args.no_path_restriction:
                    for j, dist in enumerate(self.seq_dist_list[i]):
                        if not rl_forbidden[i, 1 + j]:
                            distances[1 + j] = dist + (100 - j) * 1e-7
                    for j, cand in enumerate(ob['candidate']):
                        if not rl_forbidden[i, max_hist_len + j]:
                            distances[max_hist_len + j] = cand['distance']
                    distances[max_hist_len + len(ob['candidate'])] = ob['distance']
                else:
                    inv_path = ob['gt_path'][::-1]
                    for j, dist in enumerate(self.seq_dist_list[i]):
                        if not rl_forbidden[i, 1 + j] and self.seq_vp_list[i][j] in ob['gt_path']:
                            distances[1 + j] = inv_path.index(self.seq_vp_list[i][j])
                    for j, cand in enumerate(ob['candidate']):
                        if not rl_forbidden[i, max_hist_len + j] and cand['viewpointId'] in ob['gt_path']:
                            distances[max_hist_len + j] = inv_path.index(cand['viewpointId'])
                    if ob['viewpoint'] in ob['gt_path']:
                        distances[max_hist_len + len(ob['candidate'])] = inv_path.index(ob['viewpoint'])
                distances = np.array(distances)
                a[i] = np.argmin(distances)
        return torch.from_numpy(a).cuda()

    def _init_global_positions(self, obs, goal_pos):
        pos_cnt = self.args.gp_grid_size ** 2
        global_pos = torch.zeros((len(obs), pos_cnt, 3), dtype=torch.float).cuda()
        distances = torch.zeros(len(obs), pos_cnt, dtype=torch.float).cuda()
        for idx, ob in enumerate(obs):
            goal_x, goal_y = goal_pos[idx][0], goal_pos[idx][1]
            radius = (self.args.gp_grid_size - 1) / 2 * self.args.gp_spacing
            s_x, s_y = -radius, -radius
            for i in range(self.args.gp_grid_size):
                c_x = s_x + i * self.args.gp_spacing
                for j in range(self.args.gp_grid_size):
                    c_y = s_y + j * self.args.gp_spacing
                    global_pos[idx, i * self.args.gp_grid_size + j, 0] = c_x
                    global_pos[idx, i * self.args.gp_grid_size + j, 1] = c_y
                    distances[idx, i * self.args.gp_grid_size + j] = \
                        math.sqrt((goal_x - c_x) ** 2 + (goal_y - c_y) ** 2)
        return global_pos, distances

    def freeze_some_modules(self):
        self.fixed_modules = [self.vln_bert.vln_bert, self.vln_bert.position_encoder]
        for module in self.fixed_modules:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_some_modules(self):
        for module in self.fixed_modules:
            for param in module.parameters():
                param.requires_grad = True

    def rollout(self, train_ml=None, train_rl=True, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs(t=0)

        batch_size = len(obs)

        # Language input
        txt_ids, txt_masks, txt_lens = self._language_variable(obs)

        ''' Language BERT '''
        language_inputs = {
            'mode': 'language',
            'txt_ids': txt_ids,
            'txt_masks': txt_masks,
        }
        txt_embeds = self.vln_bert(**language_inputs)

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(obs):  # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp[0] for vp in traj[i]['path']]
            last_ndtw[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

        # Initialization the tracking state
        ended = np.array([False] * batch_size)

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.
        rl_teacher_loss = 0.
        target_predict_loss = 0.

        # for backtrack
        visited = [set() for _ in range(batch_size)]

        base_position = np.stack([ob['position'] for ob in obs], axis=0)

        goal_positions = np.stack([ob['goal_position'] for ob in obs], axis=0) - base_position[:, :2]  # bs x 2
        global_positions, goal_distances = self._init_global_positions(obs, goal_positions)
        global_pos_feat = self.vln_bert('global_pos', global_pos=global_positions, txt_embeds=txt_embeds)
        goal_pred_gt = torch.argmin(goal_distances, dim=1)

        # global embedding
        hist_embeds = self.vln_bert('history').expand(batch_size, -1).unsqueeze(1)
        hist_lens = [1 for _ in range(batch_size)]

        self._init_graph(batch_size)

        for t in range(self.args.max_action_len):
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens, ob_pos = self._cand_pano_feature_variable(obs)
                ob_masks = length2mask(ob_lens).logical_not()
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()

            ''' Visual BERT '''
            graph_mask = self._get_connectivity_mask() if t > 0 and self.args.use_conn else None
            vp_dup = torch.Tensor(self.seq_dup_vp).bool().cuda() if self.args.no_temporal and t > 0 else None
            ob_pos = ob_pos - torch.from_numpy(base_position).cuda().unsqueeze(1)  # bs x num_obs x 3
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,  # history before t step
                'hist_lens': hist_lens,
                'graph_mask': graph_mask,
                'vp_dup': vp_dup,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'ob_position': ob_pos.float(),
                'return_states': True if self.feedback == 'sample' else False,
                'global_pos_feat': global_pos_feat if self.args.global_positions else None
            }

            t_outputs = self.vln_bert(**visual_inputs)
            logit = t_outputs[0]
            hist_embeds = t_outputs[-1]
            if self.args.global_positions:
                pos_logit = t_outputs[-3]
                global_pos_feat = t_outputs[-2]
            max_hist_len = hist_embeds.size(1)
            if self.feedback == 'sample':
                h_t = t_outputs[1]
                hidden_states.append(h_t)

            # mask out logits of the current position
            if t > 0:
                logit[:, 1:max_hist_len].masked_fill_(self._get_dup_logit_mask(obs) == 0, -float('inf'))
            if self.args.no_gas:
                logit[:, 1:max_hist_len].fill_(-float('inf'))

            # reweight logits
            if self.args.logit_reweighting and not self.args.no_gas:
                for idx, ob in enumerate(obs):
                    current_vp = ob['viewpoint']
                    for jdx, hist_vp in enumerate(self.seq_vp_list[idx]):
                        logit[idx, 1 + jdx] -= self.blocked_path[idx][current_vp][hist_vp] * 1e2
                    for jdx, cand in enumerate(ob['candidate']):
                        logit[idx, max_hist_len + jdx] -= self.blocked_path[idx][current_vp][cand['viewpointId']] * 1e2

            if train_ml is not None:
                # Supervised training
                target = self._teacher_action(obs, ended, max_hist_len)
                ml_loss += self.criterion(logit, target)
            if train_rl and not self.args.no_gas:
                rl_forbidden = (logit < -1e7)
                rl_target = self._get_rl_optimal_action(obs, ended, max_hist_len, rl_forbidden)
                rl_teacher_loss += self.criterion(logit, rl_target)

            if self.args.global_positions and (train_ml is not None or train_rl):
                goal_pred_gt = goal_pred_gt.clone().detach()
                for i, end in enumerate(ended):
                    if end:
                        goal_pred_gt[i] = -100
                target_predict_loss += self.criterion(pos_logit, goal_pred_gt)

            # mask logit where the agent backtracks in observation in evaluation
            if self.args.no_cand_backtrack:  # default: skip
                bt_masks = torch.zeros(ob_nav_types.size()).bool()
                for ob_id, ob in enumerate(obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            bt_masks[ob_id][c_id] = True
                bt_masks = bt_masks.cuda()
                logit[:, max_hist_len:].masked_fill_(bt_masks, -float('inf'))


            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target  # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)  # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)  # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))  # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit / self.args.rl_temperature, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id - max_hist_len == (ob_cand_lens[i] - 1) or next_id == self.args.ignoreid or \
                        ended[i]:  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

            self._update_graph(obs, ended, cpu_a_t, max_hist_len)
            # get history input embeddings
            if train_rl or ((not np.logical_or(ended, (cpu_a_t == -1)).all()) and (t != self.args.max_action_len - 1)):
                # DDP error: RuntimeError: Expected to mark a variable ready only once.
                # It seems that every output from DDP should be used in order to perform correctly
                hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats = self._history_variable(obs)
                prev_act_angle = np.zeros((batch_size, self.args.angle_feat_size), np.float32)
                for i, next_id in enumerate(cpu_a_t):
                    if next_id != -1:
                        if next_id >= max_hist_len:
                            prev_act_angle[i] = \
                                obs[i]['candidate'][next_id - max_hist_len]['feature'][-self.args.angle_feat_size:]
                prev_act_angle = torch.from_numpy(prev_act_angle).cuda()

                position = np.stack([ob['position'] for ob in obs], axis=0) - base_position  # bs x 3
                position = torch.from_numpy(position).cuda().float()

                t_hist_inputs = {
                    'mode': 'history',
                    'hist_img_feats': hist_img_feats,
                    'hist_ang_feats': prev_act_angle,
                    'hist_pano_img_feats': hist_pano_img_feats,
                    'hist_pano_ang_feats': hist_pano_ang_feats,
                    'ob_step': t if not self.args.no_temporal else None,
                    'position': position,
                }
                t_hist_embeds = self.vln_bert(**t_hist_inputs)
                hist_embeds = torch.cat([hist_embeds, t_hist_embeds.unsqueeze(1)], dim=1)

                for i, i_ended in enumerate(ended):
                    if not i_ended:
                        hist_lens[i] += 1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, max_hist_len, traj=traj)
            obs = self.env._get_obs(t=t+1)

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(obs):
                    dist[i] = ob['distance']
                    path_act = [vp[0] for vp in traj[i]['path']]
                    ndtw_score[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:  # If the action now is end
                            if dist[i] < 3.0:  # Correct
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:  # Incorrect
                                reward[i] = -2.0
                        else:  # The action is not end
                            # Path fidelity rewards (distance & nDTW)
                            reward[i] = - (dist[i] - last_dist[i])  # this distance is not normalized
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:  # Quantification
                                reward[i] = 1.0 + ndtw_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0 + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens, ob_pos = self._cand_pano_feature_variable(obs)
                ob_masks = length2mask(ob_lens).logical_not()
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()

            ''' Visual BERT '''
            graph_mask = self._get_connectivity_mask() if self.args.use_conn else None
            ob_pos = ob_pos - torch.from_numpy(base_position).cuda().unsqueeze(1)
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,
                'hist_lens': hist_lens,
                'graph_mask': graph_mask,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'ob_position': ob_pos.float(),
                'return_states': True,
                'global_pos_feat': global_pos_feat if self.args.global_positions else None
            }
            temp_output = self.vln_bert(**visual_inputs)
            last_h_ = temp_output[1]

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()  # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:  # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length - 1, -1, -1):
                discount_reward = discount_reward * self.args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = torch.from_numpy(masks[t]).cuda()
                clip_reward = discount_reward.copy()
                r_ = torch.from_numpy(clip_reward).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                t_policy_loss = (-policy_log_probs[t] * a_ * mask_).sum()
                t_critic_loss = (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss

                rl_loss += t_policy_loss + t_critic_loss
                if self.feedback == 'sample':
                    rl_loss += (- self.args.entropy_loss_weight * entropys[t] * mask_).sum()

                self.logs['critic_loss'].append(t_critic_loss.item())
                self.logs['policy_loss'].append(t_policy_loss.item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if self.args.normalize_loss == 'total':
                rl_loss /= total
            elif self.args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert self.args.normalize_loss == 'none'

            if self.args.rl_teacher_only:
                rl_loss = rl_teacher_loss * self.args.rl_teacher_weight / batch_size
            else:
                rl_loss += rl_teacher_loss * self.args.rl_teacher_weight / batch_size
            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())  # critic loss + policy loss + entropy loss

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if self.args.global_positions and (train_rl or train_ml is not None):
            self.loss += target_predict_loss * self.args.gp_loss_weight / batch_size
            self.logs['GP_loss'].append((target_predict_loss * self.args.gp_loss_weight / batch_size).item())
        elif not self.args.global_positions:
            self.logs['GP_loss'].append(0)

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.

        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        """ Evaluate once on each instruction in the current environment """
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        super().test(iters=iters)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=self.args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

        self.vln_bert_optimizer.step()
        self.critic_optimizer.step()

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.vln_bert.train()
        self.critic.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0

            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=self.args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':  # agents in IL and RL separately
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            # print(self.rank, iter, self.loss)
            self.loss.backward()

            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()

            if self.args.aug is None:
                print_progress(iter, n_iters + 1, prefix='Progress:', suffix='Complete', bar_length=50)

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}

        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            state_dict = states[name]['state_dict']
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
                if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            state.update(state_dict)
            model.load_state_dict(state)
            if self.args.resume_optimizer:
                optimizer.load_state_dict(states[name]['optimizer'])

        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1
