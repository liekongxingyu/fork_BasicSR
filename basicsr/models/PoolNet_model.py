import os
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

import torch.nn.functional as F
import torch.nn as nn


from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class PoolNetModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(PoolNetModel, self).__init__(opt)

        self.save_vis = opt["val"]["save_vis"]  # 是否保存可视化结果

        self.save_vis_freq = opt["val"].get("save_vis_freq")  # 可视化结果保存频率
        if self.save_vis_freq is None:
            self.save_vis_freq = int(opt['val']['val_freq'])

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        # 是否打印网络架构
        # self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
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
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)

        self.cri_pix = nn.L1Loss()

        # 列表输出：保持原有逻辑
        self.output = preds[-1]
        self.sr = preds[-1]


        preds_for_loss = preds

        label_img = self.gt

        label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
        label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')

        l1 = self.cri_pix(preds[0], label_img4)
        l2 = self.cri_pix(preds[1], label_img2)
        l3 = self.cri_pix(preds[2], label_img)

        loss_content = l1+l2+l3

        label_fft1 = torch.fft.fft2(label_img4, dim=(-2,-1))
        label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

        pred_fft1 = torch.fft.fft2(preds[0], dim=(-2,-1))
        pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

        label_fft2 = torch.fft.fft2(label_img2, dim=(-2,-1))
        label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

        pred_fft2 = torch.fft.fft2(preds[1], dim=(-2,-1))
        pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

        label_fft3 = torch.fft.fft2(label_img, dim=(-2,-1))
        label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

        pred_fft3 = torch.fft.fft2(preds[2], dim=(-2,-1))
        pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)


        f1 = self.cri_pix(pred_fft1, label_fft1)
        f2 = self.cri_pix(pred_fft2, label_fft2)
        f3 = self.cri_pix(pred_fft3, label_fft3)
        loss_fft = f1+f2+f3


        loss = loss_content + 0.1 * loss_fft
        loss.backward()

        self.optimizer_g.step()


        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


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
                    # 动态保存所有中间特征（只保存第一个batch用于可视化）
                    if i == 0:
                        for key, value in pred.items():
                            if key != 'output':
                                setattr(self, key, value)
                                
                elif isinstance(pred, list):
                    # 处理列表输出
                    pred = pred[-1]
                    outs.append(pred.detach().cpu())
                else:
                    # 处理单tensor输出
                    outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()


    # 这个函数是用来更新网络参数的指数移动平均值
    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    # 非分布式验证全流程
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, use_pbar=True, use_image=False):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        # 安全获取 max_val_num
        max_val_num = getattr(self, 'max_val_num', float('inf'))
        total_samples = len(dataloader)
        actual_samples = min(total_samples, max_val_num) 

        metric_data = dict()
        
        # 修复：条件创建进度条，使用正确的总数
        if use_pbar:
            pbar = tqdm(total=actual_samples, unit='image')

        visualization_data = []
        max_vis_samples = 5

        processed_count = 0  # 添加实际处理计数器
        
        for idx, val_data in enumerate(dataloader):
            if idx >= actual_samples:
                break

            img_name = osp.splitext(osp.basename(val_data['lq_path']))
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            if len(visualization_data) < max_vis_samples:
                vis_data = {
                    'img_name': img_name,
                    'visuals': visuals.copy()
                }
                visualization_data.append(vis_data)

            # GPU 内存清理
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            # 保存图像逻辑（保持不变）
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                f'{img_name}_{self.opt["name"]}.png')
                
                # 调试信息
                # print(f"图像数据: {sr_img.shape if sr_img is not None else 'None'}")
                # print(f"保存路径: {save_img_path}")
                # print(f"目录存在: {osp.exists(osp.dirname(save_img_path))}")
                
                # 确保目录存在
                os.makedirs(osp.dirname(save_img_path), exist_ok=True)
                
                # 保存并检查结果
                success = imwrite(sr_img, save_img_path)
                # print(f"保存{'成功' if success else '失败'}")

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            
            processed_count += 1
            
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        
        if use_pbar:
            pbar.close()

        if with_metrics:
            # 修复：使用实际处理的样本数
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= processed_count
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        
        self.stage_visualization(visualization_data, current_iter)


    def stage_visualization(self, visualization_data, current_iter):
        pass

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
