# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
import os
import numpy as np
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY


loss_module = importlib.import_module('basicsr.losses')
metric_module = importlib.import_module('basicsr.metrics')

@MODEL_REGISTRY.register()
class NAFNetModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(NAFNetModel, self).__init__(opt)

        self.max_val_num = opt['val'].get('max_val_num', None)  # 最大验证图像数量
    
        self.save_vis = opt["val"]["save_vis"]  # 是否保存可视化结果
        self.save_vis_freq = opt["val"].get("save_vis_freq", 10)  # 可视化结果保存频率

        # define network
        self.net_g = build_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(
                train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
        #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
        #             optim_params_lowlr.append(v)
        #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)


        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        #adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i//scale*scale
        step_j = step_j//scale*scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(self.lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()

        preds = self.net_g(self.lq)
        
        # 🔧 修复：统一处理网络输出格式
        if isinstance(preds, dict):
            # 字典输出：取主要输出用于损失计算
            self.output = preds['output'] 
            self.sr = preds['output']
            # 保存中间结果用于可视化
            self.feature1 = preds.get('feature1', None)
            self.feature2 = preds.get('feature2', None)
            # 用于损失计算的输出列表
            preds_for_loss = [preds['output']]
        elif isinstance(preds, list):
            # 列表输出：保持原有逻辑
            self.output = preds[-1]
            self.sr = preds[-1] 
            preds_for_loss = preds
        else:
            # 单tensor输出
            self.output = preds
            self.sr = preds
            preds_for_loss = [preds]

        l_total = 0
        loss_dict = OrderedDict()
        
        # pixel loss - 使用修复后的逻辑
        if self.cri_pix:
            l_pix = 0.
            for pred in preds_for_loss:  # 🔧 使用处理后的列表
                l_pix += self.cri_pix(pred, self.gt)
            
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss - 使用self.sr确保一致性
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.sr, self.gt)  # 🔧 使用self.sr
            
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)


    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred = self.net_g(self.lq[i:j])
                if isinstance(pred, dict):
                    # 处理字典输出
                    outs.append(pred['output'].detach().cpu())
                    # 保存特征（只保存第一个batch的特征用于可视化）
                    if i == 0:
                        self.feature1 = pred.get('feature1', None)
                        self.feature2 = pred.get('feature2', None)
                elif isinstance(pred, list):
                    pred = pred[-1]
                    outs.append(pred.detach().cpu())
                else:
                    outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr=True, use_image=True):
        """真正的非分布式验证实现"""
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        max_val_num = self.max_val_num
        total_samples = len(dataloader)
        actual_samples = min(total_samples, max_val_num) 

        logger = get_root_logger()
        if max_val_num < total_samples:
            logger.info(f'Validation limited to {actual_samples}/{total_samples} samples')


        # 🔧 非分布式：直接创建进度条，不需要rank判断
        pbar = tqdm(total=len(dataloader), unit='image')
        cnt = 0

        # 存储可视化数据
        visualization_data = []
        max_vis_samples = 5

        # 🔧 非分布式：遍历所有数据，不需要分布式任务分配
        for idx, val_data in enumerate(dataloader):
            if idx >= actual_samples:
                break

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # 收集可视化数据
            if len(visualization_data) < max_vis_samples:
                vis_data = {
                    'img_name': img_name,
                    'visuals': visuals.copy()
                }
                visualization_data.append(vis_data)

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]
                    visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)
                    imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                else:
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                            img_name, f'{img_name}_{current_iter}.png')
                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name, f'{img_name}_{current_iter}_gt.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, f'{img_name}_gt.png')

                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            # 🔧 非分布式：直接更新进度条
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        
        pbar.close()

        # 🔧 关键修复：非分布式的指标计算（不使用torch.distributed）
        if with_metrics:
            metrics_dict = {}
            for key in self.metric_results:
                metrics_dict[key] = self.metric_results[key] / cnt  # 直接计算平均值

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger, metrics_dict)
        
        # 可视化处理
        self.stage_visualization(visualization_data, current_iter)
        return 0.


    def stage_visualization(self, visualization_data, current_iter):
        # 🆕 新增：集成的可视化功能
        if self.save_vis and current_iter % self.save_vis_freq == 0:  # 默认每10次迭代可视化一次
            vis_save_dir = osp.join(
                self.opt['path']['visualization'], f'feature_analysis_{current_iter}')
            self._visualize_frequency_decomposition(
                visualization_data, vis_save_dir)
            
    def _visualize_frequency_decomposition(self, visualization_data, save_dir):
        """集成的特征可视化函数"""
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        os.makedirs(save_dir, exist_ok=True)
        logger = get_root_logger()
        logger.info(f'Generating feature visualization in {save_dir}')
        
        for idx, vis_data in enumerate(visualization_data):
            img_name = vis_data['img_name']
            visuals = vis_data['visuals']
            
            try:
                # 转换为numpy格式
                lq_img = tensor2img(visuals['lq'])
                result_img = tensor2img(visuals['result'])
                
                gt_img = None
                if 'gt' in visuals:
                    gt_img = tensor2img(visuals['gt'])
                
                # 检查是否有两个特征的能量数据
                if 'feature1_energy' in visuals and 'feature2_energy' in visuals:
                    feature1_energy = tensor2img(visuals['feature1_energy'])
                    feature2_energy = tensor2img(visuals['feature2_energy'])
                    
                    # 检查是否有特征差异图
                    feature_diff = None
                    if 'feature_diff_energy' in visuals:
                        feature_diff = tensor2img(visuals['feature_diff_energy'])
                    
                    # 创建特征对比图布局 (2x3 或 2x2)
                    if feature_diff is not None:
                        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                        has_diff = True
                    else:
                        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                        has_diff = False
                    
                    # 第一行：输入输出对比
                    axes[0, 0].imshow(lq_img)
                    axes[0, 0].set_title('输入图像', fontsize=14, weight='bold')
                    axes[0, 0].axis('off')
                    
                    axes[0, 1].imshow(result_img)
                    axes[0, 1].set_title('输出结果', fontsize=14, weight='bold')
                    axes[0, 1].axis('off')
                    
                    if has_diff and gt_img is not None:
                        axes[0, 2].imshow(gt_img)
                        axes[0, 2].set_title('真实标签', fontsize=14, weight='bold')
                        axes[0, 2].axis('off')
                    elif has_diff:
                        axes[0, 2].axis('off')
                    
                    # 第二行：特征能量对比
                    im1 = axes[1, 0].imshow(feature1_energy, cmap='hot', interpolation='bilinear')
                    axes[1, 0].set_title('Middle Blocks 前\n特征能量分布', fontsize=14, weight='bold')
                    axes[1, 0].axis('off')
                    plt.colorbar(im1, ax=axes[1, 0], shrink=0.6, label='能量强度')
                    
                    im2 = axes[1, 1].imshow(feature2_energy, cmap='hot', interpolation='bilinear')
                    axes[1, 1].set_title('Middle Blocks 后\n特征能量分布', fontsize=14, weight='bold')
                    axes[1, 1].axis('off')
                    plt.colorbar(im2, ax=axes[1, 1], shrink=0.6, label='能量强度')
                    
                    if has_diff:
                        im3 = axes[1, 2].imshow(feature_diff, cmap='RdBu_r', interpolation='bilinear')
                        axes[1, 2].set_title('特征变化差异图\n(红色=增强, 蓝色=减弱)', fontsize=14, weight='bold')
                        axes[1, 2].axis('off')
                        plt.colorbar(im3, ax=axes[1, 2], shrink=0.6, label='差异强度')
                    
                    # 添加整体标题和统计信息
                    plt.suptitle(f'Middle Blocks 特征分析 - {img_name}', fontsize=16, y=0.95, weight='bold')
                    
                    # 在底部添加简单的统计信息
                    feat1_stats = f"前: 均值={np.mean(feature1_energy):.3f}, 最大={np.max(feature1_energy):.3f}"
                    feat2_stats = f"后: 均值={np.mean(feature2_energy):.3f}, 最大={np.max(feature2_energy):.3f}"
                    plt.figtext(0.5, 0.02, f"{feat1_stats} | {feat2_stats}", 
                            ha='center', fontsize=10, 
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
                    
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.90, bottom=0.08)
                    
                    save_path = f"{save_dir}/{img_name}_feature_analysis.png"
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f'Saved feature analysis for {img_name}')
                
                else:
                    logger.warning(f'No feature energy data available for {img_name}')
                    
            except Exception as e:
                logger.error(f'Error creating visualization for {img_name}: {str(e)}')
                continue


    def get_current_visuals(self):
        """获取当前的可视化结果"""
        
        out_dict = OrderedDict()
        
        # 基础数据
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        
        # 处理两个特征的能量可视化
        if hasattr(self, 'feature1') and hasattr(self, 'feature2'):
            if self.feature1 is not None:
                # 计算feature1的能量图
                feat1_energy = self.feature1.pow(2).mean(1, keepdim=True).detach().cpu()
                # 归一化
                feat1_energy = (feat1_energy - feat1_energy.min()) / (feat1_energy.max() - feat1_energy.min() + 1e-8)
                # 扩展为3通道
                out_dict['feature1_energy'] = feat1_energy.repeat(1, 3, 1, 1)
            
            if self.feature2 is not None:
                feat2_energy = self.feature2.pow(2).mean(1, keepdim=True).detach().cpu()
                feat2_energy = (feat2_energy - feat2_energy.min()) / (feat2_energy.max() - feat2_energy.min() + 1e-8)
                out_dict['feature2_energy'] = feat2_energy.repeat(1, 3, 1, 1)
            
            # 特征差异能量图
            if self.feature1 is not None and self.feature2 is not None:
                feat_diff = (self.feature2 - self.feature1).detach().cpu()
                diff_energy = feat_diff.pow(2).mean(1, keepdim=True)
                diff_energy = (diff_energy - diff_energy.min()) / (diff_energy.max() - diff_energy.min() + 1e-8)
                out_dict['feature_diff_energy'] = diff_energy.repeat(1, 3, 1, 1)
        
        return out_dict


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict


    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
