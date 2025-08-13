import os
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

@MODEL_REGISTRY.register()
class LightweightDerainModel(BaseModel):
    """轻量级去雨模型"""

    def __init__(self, opt):
        super(LightweightDerainModel, self).__init__(opt)

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
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
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
        self.output = self.net_g(self.lq)

        if isinstance(self.output, dict):
            self.sr = self.output['output']
            # 保存中间结果用于可视化
            self.low_freq_feat = self.output.get('low_freq_feat', None)
            self.high_freq_feat = self.output.get('high_freq_feat', None)
        else:
            self.sr = self.output

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss - 修复这里的bug
        if self.cri_pix:
            l_pix = self.cri_pix(self.sr, self.gt)  # 改为self.sr
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss - 修复这里的bug
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.sr, self.gt)  # 改为self.sr
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        """测试函数"""
        if hasattr(self, 'net_g_ema') and self.net_g_ema:
            self.net_g_ema.apply_shadow()
        
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
            
            # 处理输出格式，保存详细信息
            if isinstance(self.output, dict):
                self.sr = self.output['output']
                # 保存详细信息用于可视化
                self.test_results = self.output
            else:
                self.sr = self.output
                self.test_results = {'output': self.sr}
        
        # 修复：恢复网络状态
        if hasattr(self, 'net_g_ema') and self.net_g_ema:
            self.net_g_ema.restore()
        self.net_g.train()

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
                out_list = [self.net_g(aug) for aug in lq_list]  # 修复：改为self.net_g
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

    # 分布式验证全流程 - 集成可视化功能
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        # 存储可视化数据
        visualization_data = []
        max_vis_samples = 5  # 最多可视化5张图

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img

            # 收集可视化数据
            if len(visualization_data) < max_vis_samples:
                vis_data = {
                    'img_name': img_name,
                    'visuals': visuals.copy()  # 深拷贝避免后续删除影响
                }
                visualization_data.append(vis_data)

            # 清理内存
            if hasattr(self, 'gt'):
                del self.gt
            del self.lq
            del self.sr  # 修复：改为self.sr
            torch.cuda.empty_cache()

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
                
                print(f"保存路径: {save_img_path}")
                # 确保目录存在
                os.makedirs(osp.dirname(save_img_path), exist_ok=True)
                # 保存并检查结果
                success = imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        
        # 🆕 新增：集成的可视化功能
        if save_img and current_iter % 5000 == 0:  # 每5000次迭代可视化一次
            vis_save_dir = osp.join(self.opt['path']['visualization'], f'frequency_analysis_{current_iter}')
            self._visualize_frequency_decomposition(visualization_data, vis_save_dir)

    def _visualize_frequency_decomposition(self, visualization_data, save_dir):
        """集成的频率分解可视化函数"""
        os.makedirs(save_dir, exist_ok=True)
        logger = get_root_logger()
        logger.info(f'Generating frequency decomposition visualization in {save_dir}')
        
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
                
                # 如果有频率分解结果
                if 'low_freq_energy' in visuals and 'high_freq_energy' in visuals:
                    low_energy_img = tensor2img(visuals['low_freq_energy'])
                    high_energy_img = tensor2img(visuals['high_freq_energy'])
                    
                    # 创建完整的对比图
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    
                    axes[0,0].imshow(lq_img)
                    axes[0,0].set_title('含雨图像')
                    axes[0,0].axis('off')
                    
                    axes[0,1].imshow(result_img)
                    axes[0,1].set_title('去雨结果')
                    axes[0,1].axis('off')
                    
                    if gt_img is not None:
                        axes[0,2].imshow(gt_img)
                        axes[0,2].set_title('真实清晰图')
                        axes[0,2].axis('off')
                    else:
                        axes[0,2].axis('off')
                    
                    axes[1,0].imshow(low_energy_img, cmap='hot')
                    axes[1,0].set_title('低频能量图\n(背景结构)')
                    axes[1,0].axis('off')
                    
                    axes[1,1].imshow(high_energy_img, cmap='hot')
                    axes[1,1].set_title('高频能量图\n(细节/雨纹)')
                    axes[1,1].axis('off')
                    
                    # 显示学到的滤波器权重
                    if hasattr(self.net_g, 'lowpass_filter'):
                        filter_weight = self.net_g.lowpass_filter.weight[0,0].detach().cpu().numpy()
                        axes[1,2].imshow(filter_weight, cmap='RdBu')
                        axes[1,2].set_title('学到的低通滤波器')
                        axes[1,2].axis('off')
                    else:
                        axes[1,2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(f"{save_dir}/{img_name}_frequency_analysis.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f'Saved frequency analysis for {img_name}')
                else:
                    logger.warning(f'No frequency data available for {img_name}')
                    
            except Exception as e:
                logger.error(f'Error creating visualization for {img_name}: {str(e)}')
                continue

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
        """获取当前的可视化结果"""
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.sr.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        
        # 频率分解结果的可视化处理
        if hasattr(self, 'test_results') and isinstance(self.test_results, dict):
            if 'low_freq_feat' in self.test_results and self.test_results['low_freq_feat'] is not None:
                # 将特征图转换为可视化格式
                low_feat = self.test_results['low_freq_feat'].detach().cpu()
                high_feat = self.test_results['high_freq_feat'].detach().cpu()
                
                # 计算能量图并归一化
                low_energy = low_feat.pow(2).mean(1, keepdim=True)  # [B,1,H,W]
                high_energy = high_feat.pow(2).mean(1, keepdim=True)
                
                # 归一化到[0,1]
                low_energy = (low_energy - low_energy.min()) / (low_energy.max() - low_energy.min() + 1e-8)
                high_energy = (high_energy - high_energy.min()) / (high_energy.max() - high_energy.min() + 1e-8)
                
                # 扩展到3通道用于保存（BasicSR框架要求）
                out_dict['low_freq_energy'] = low_energy.repeat(1, 3, 1, 1)
                out_dict['high_freq_energy'] = high_energy.repeat(1, 3, 1, 1)
        
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
