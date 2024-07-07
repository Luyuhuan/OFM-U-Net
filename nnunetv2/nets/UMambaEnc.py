import numpy as np
import torch
from torch import nn
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba

lzz_flag = 1

import cv2
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
def flow_to_color(flow, max_flow=None):
    # 使用极坐标转换光流的(x, y)到(magnitude, angle)
    magnitude, angle = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))

    # 标准化亮度（V）和饱和度（S）来反映光流的大小
    if max_flow is not None:
        magnitude = np.clip(magnitude / max_flow, 0, 1)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    
    # 角度是0-360度，对应色调的0-180度
    hsv[..., 0] = angle * (180 / np.pi / 2)
    
    # 设定饱和度为最大值
    hsv[..., 1] = 255

    # 标准化亮度值到[0,255]
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    # 转换HSV到BGR以显示
    color_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return color_flow

def draw_flow(img, flow, step=48):
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
    fx, fy = flow[y, x].T

    # 创建线条终点坐标
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    # 创建图像并绘制线条
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, isClosed=False, color=(0, 255, 0))
    
    # 为每个线条绘制一个小圆点
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def compute_flow(args):
    frame1, frame2 = args
    frame1 = frame1.numpy()
    frame2 = frame2.numpy()

    # OpenCV期望的是uint8，shape为[H, W, C]
    frame1 = np.moveaxis(frame1, 0, -1)
    frame2 = np.moveaxis(frame2, 0, -1)

    # 转换为灰度图像
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    return flow

def compute_flow_and_resize(args):
    frame1, frame2, size = args
    frame1 = frame1.numpy()
    frame2 = frame2.numpy()

    # OpenCV期望的是uint8，shape为[H, W, C]
    frame1 = np.moveaxis(frame1, 0, -1)
    frame2 = np.moveaxis(frame2, 0, -1)

    # 转换为灰度图像
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 调整光流尺寸
    flow_resized = cv2.resize(flow, (size, size))
    
    return flow_resized

def parallel_flow_computation(video_frames_tensor, target_sizes, orgflag):
    # 处理batch_size大于1的数据
    batch_size, num_frames, c, h, w = video_frames_tensor.size()
    # 不再需要断言批次大小为1

    # 准备并行任务
    tasks = []
    for batch_idx in range(batch_size):
        for frame_idx in range(num_frames - 1):
            frame1 = video_frames_tensor[batch_idx, frame_idx]
            frame2 = video_frames_tensor[batch_idx, frame_idx + 1]
            if orgflag == 1:
                # 可以根据需要调整任务格式，这里假设compute_flow需要frame1和frame2作为输入
                tasks.append((frame1, frame2))
    # 使用进程池并行计算光流
    with ProcessPoolExecutor() as executor:
        flows = list(executor.map(compute_flow, tasks))

    # 重新组织flows以匹配batch_size和num_frames
    # 假设每个流计算结果是单独的，需要将它们重新组织成(batch_size, num_frames-1, ...)的形式
    flows_reshaped = []
    per_batch_task_count = num_frames - 1
    for i in range(0, len(flows), per_batch_task_count):
        flows_reshaped.append(flows[i:i+per_batch_task_count])

    return flows_reshaped

    batch_size, num_frames, c, h, w = video_frames_tensor.size()
    assert batch_size == 1, "Batch size should be 1."
    video_frames_tensor = video_frames_tensor.squeeze(0)  # 去掉批次维度
    if orgflag == 1:
        tasks = [(video_frames_tensor[i], video_frames_tensor[i+1]) for i in range(num_frames - 1)]
        with ProcessPoolExecutor() as executor:
            flows = list(executor.map(compute_flow, tasks))
        return flows
    else:
        # 准备参数
        tasks = [(video_frames_tensor[i], video_frames_tensor[i+1], size) for i in range(num_frames - 1) for size in target_sizes]

        # 使用ProcessPoolExecutor来并行处理
        flows_resized = []
        with ProcessPoolExecutor() as executor:
            flows_resized = list(executor.map(compute_flow_and_resize, tasks))

        # 将结果按尺寸组织
        flows_by_size = {size: [] for size in target_sizes}
        for i, flow in enumerate(flows_resized):
            size_index = i % len(target_sizes)
            flows_by_size[target_sizes[size_index]].append(flow)
        
        return flows_by_size

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
    
    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2).contiguous()
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).contiguous().reshape(B, C, *img_dims)

        return out


class ResidualMambaEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 return_skips: bool = False,
                 disable_default_stem: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if bottleneck_channels is None or isinstance(bottleneck_channels, int):
            bottleneck_channels = [bottleneck_channels] * n_stages
        assert len(
            bottleneck_channels) == n_stages, "bottleneck_channels must be None or have as many entries as we have resolution stages (n_stages)"
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        # build a stem, Todo maybe we need more flexibility for this in the future. For now, if you need a custom
        #  stem you can just disable the stem and build your own.
        #  THE STEM DOES NOT DO STRIDE/POOLING IN THIS IMPLEMENTATION
        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            self.stem = StackedConvBlocks(1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1, conv_bias,
                                          norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
            input_channels = stem_channels
        else:
            self.stem = None

        # now build the network
        stages = []
        mamba_layers = []
        # features_per_stage = [32, 64, 128, 256, 512, 320]
        features_per_stage = [32, 64, 128, 128, 64, 32]
        for s in range(n_stages):
            stride_for_conv = strides[s] if pool_op is None else 1

            stage = StackedResidualBlocks(
                n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], stride_for_conv,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                block=block, bottleneck_channels=bottleneck_channels[s], stochastic_depth_p=stochastic_depth_p,
                squeeze_excitation=squeeze_excitation,
                squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
            )

            if pool_op is not None:
                stage = nn.Sequential(pool_op(strides[s]), stage)

            stages.append(stage)
            input_channels = features_per_stage[s]

            mamba_layers.append(MambaLayer(input_channels))

        #self.stages = nn.Sequential(*stages)
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

        self.mamba_layers = nn.ModuleList(mamba_layers)

    def forward(self, x):
        global lzz_flag
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        #for s in self.stages:
        for s in range(len(self.stages)):
            #x = s(x)
            # print(self.stages[s])
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output

class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualMambaEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        global lzz_flag
        if lzz_flag == 1:
            # 有效的结构 256 * 256 *************************************************
            lzz_kernelsize = [[(3,3,3), (3,3,3), (3,3,3)],
                              [(3,3,3), (3,3,3), (3,3,3), (3,3,3)],
                              [(3,3,3), (3,3,3), (3,3,3), (3,3,3)],
                              [(3,3,3), (3,3,3), (3,3,3), (3,3,3)],
                              [(3,3,3), (3,3,3), (3,3,3), (3,3,3), (3,3,3)]]
            lzz_initial_stride = [[(2,2,2), (2,1,1), (2,1,1)],
                                  [(2,2,2), (2,2,2), (2,1,1), (2,1,1)],
                                  [(2,2,2), (2,2,2), (2,2,2), (2,1,1)],
                                  [(2,2,2), (2,2,2), (2,2,2), (1,2,2)],
                                  [(2,2,2), (2,2,2), (1,2,2), (1,2,2), (1,2,2)]]
            # 有效的结构 256 * 256 *************************************************

            # size 调整为 128
            # lzz_initial_stride = [[(2,1,1), (2,1,1), (2,1,1)],
            #                       [(2,2,2), (2,1,1), (2,1,1), (2,1,1)],
            #                       [(2,2,2), (2,2,2), (2,1,1), (2,1,1)],
            #                       [(2,2,2), (2,2,2), (2,2,2), (1,1,1)],
            #                       [(2,2,2), (2,2,2), (1,2,2), (1,2,2), (1,1,1)]]

            # 有效的结构 256 * 256 *************************************************
            # lzzfcio = [[[8*8*8, 2*8*8], [2*8*8, 2*2*8], [2*2*8, 1]],
            #            [[8*8*8, 2*8*8], [2*8*8, 2*2*8], [2*2*8, 1]],
            #            [[8*8*8, 2*8*8], [2*8*8, 2*2*8], [2*2*8, 1]],
            #            [[8*8*8, 2*8*8], [2*8*8, 2*2*8], [2*2*8, 1]],
            #            [[8*8*8, 2*8*8], [2*8*8, 2*2*8], [2*2*8, 1]]]
            # 有效的结构 256 * 256 *************************************************
            
            # size 调整为 192 * 192
            lzzfcio = [[[8*6*6, 2*6*6], [2*6*6, 2*3*3], [2*3*3, 1]],
                       [[8*6*6, 2*6*6], [2*6*6, 2*3*3], [2*3*3, 1]],
                       [[8*6*6, 2*6*6], [2*6*6, 2*3*3], [2*3*3, 1]],
                       [[8*6*6, 2*6*6], [2*6*6, 2*3*3], [2*3*3, 1]],
                       [[8*6*6, 2*6*6], [2*6*6, 2*3*3], [2*3*3, 1]]]
            # 添加 class head
            
            # lzzclassio = [[[8*6*6, 2*6*6], [2*6*6, 2*3*3], [2*3*3, 4]],
            #            [[8*6*6, 2*6*6], [2*6*6, 2*3*3], [2*3*3, 4]],
            #            [[8*6*6, 2*6*6], [2*6*6, 2*3*3], [2*3*3, 4]],
            #            [[8*6*6, 2*6*6], [2*6*6, 2*3*3], [2*3*3, 4]],
            #            [[8*6*6, 2*6*6], [2*6*6, 2*3*3], [2*3*3, 4]]]

            # lzzclassio = [[[8*6*6, 2*6*6], [2*6*6, 2*3*3], [2*3*3, 2]],
            #             [[8*6*6, 2*6*6], [2*6*6, 2*3*3], [2*3*3, 2]],
            #             [[8*6*6, 2*6*6], [2*6*6, 2*3*3], [2*3*3, 2]],
            #             [[8*6*6, 2*6*6], [2*6*6, 2*3*3], [2*3*3, 2]],
            #             [[8*6*6, 2*6*6], [2*6*6, 2*3*3], [2*3*3, 2]]]
            
            # class head只在最后一层
            lzzclassio = [[[2*3*3, 2]],
                        [[2*3*3, 2]],
                        [[2*3*3, 2]],
                        [[2*3*3, 2]],
                        [[2*3*3, 2]]]
            # lzzclassio = [[[2*3*3, 4]],
            #             [[2*3*3, 4]],
            #             [[2*3*3, 4]],
            #             [[2*3*3, 4]],
            #             [[2*3*3, 4]]]
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        lzz_layers = []
        lzz_fc = []
        lzz_classhead = []
        # 新增加
        # n_stages_encoder = 2
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedResidualBlocks(
                n_blocks = n_conv_per_stage[s-1],
                conv_op = encoder.conv_op,
                input_channels = 2 * input_features_skip,
                output_channels = input_features_skip,
                kernel_size = encoder.kernel_sizes[-(s + 1)],
                initial_stride = 1,
                conv_bias = encoder.conv_bias,
                norm_op = encoder.norm_op,
                norm_op_kwargs = encoder.norm_op_kwargs,
                dropout_op = encoder.dropout_op,
                dropout_op_kwargs = encoder.dropout_op_kwargs,
                nonlin = encoder.nonlin,
                nonlin_kwargs = encoder.nonlin_kwargs,
            ))
            if lzz_flag == 1:
                nowlzzlayer = []
                for nowks, nowis in zip(lzz_kernelsize[s-1], lzz_initial_stride[s-1]):
                    nowlzzlayer.append(StackedResidualBlocks(
                        n_blocks = n_conv_per_stage[s-1],
                        conv_op = encoder.conv_op,
                        # input_channels = 48,
                        # output_channels = 48,
                        
                        input_channels = 40,
                        output_channels = 40,
                        kernel_size = nowks,
                        initial_stride = nowis,
                        conv_bias = encoder.conv_bias,
                        norm_op = encoder.norm_op,
                        norm_op_kwargs = encoder.norm_op_kwargs,
                        dropout_op = encoder.dropout_op,
                        dropout_op_kwargs = encoder.dropout_op_kwargs,
                        nonlin = encoder.nonlin,
                        nonlin_kwargs = encoder.nonlin_kwargs,
                    ))
                nowlzzlayer = nn.ModuleList(nowlzzlayer)
                lzz_layers.append(nowlzzlayer)

                lzzfc = []
                # Tanhflag = 1
                # for nowio in lzzfcio[s-1]:
                #     if Tanhflag == 1:
                #         lzzfc.append(nn.Sequential( nn.Dropout(0.01, ), nn.Linear(nowio[0], nowio[1]), nn.Tanh()))
                #         Tanhflag = 0
                #     else:
                #         lzzfc.append(nn.Sequential( nn.Dropout(0.01, ), nn.Linear(nowio[0], nowio[1]), nn.ReLU()))
                #         Tanhflag = 1
                for nowio in lzzfcio[s-1]:
                    # if nowio == [2*8*8, 2*2*8]:
                    #     lzzfc.append(nn.Sequential(nn.Linear(nowio[0], nowio[1]), nn.Tanh(), nn.Dropout(0.2, )))
                    # else:
                    #     lzzfc.append(nn.Sequential(nn.Linear(nowio[0], nowio[1]), nn.Tanh()))
                    lzzfc.append(nn.Sequential(nn.Linear(nowio[0], nowio[1]), nn.Tanh()))
                lzzfc = nn.ModuleList(lzzfc)
                lzz_fc.append(lzzfc)

                
                lzzclasshead = []
                for nowio in lzzclassio[s-1]:
                    lzzclasshead.append(nn.Sequential(nn.Linear(nowio[0], nowio[1]), nn.Tanh()))
                lzzclasshead = nn.ModuleList(lzzclasshead)
                lzz_classhead.append(lzzclasshead)
            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))
        # lzzfcio = [[32*4*4, 16*4*4], [16*4*4, 8*4*4], [8*4*4, 1]]
        # lzzfcio = [[40*4*4, 20*4*4], [20*4*4, 10*4*4], [10*4*4, 1]]
        # lzzfcio = [[40*8*8, 10*8*8], [10*8*8, 10*4*4], [10*4*4, 1]]
        # lzzfcio = [[32*8*8, 10*8*8], [10*8*8, 10*4*4], [10*4*4, 1]]
        lzzfcio = [[8*8*8, 2*8*8], [2*8*8, 2*2*8], [2*2*8, 1]]
        lzzonefc = []
        for nowio in lzzfcio:
            # lzzonefc.append(nn.Sequential(nn.Linear(nowio[0], nowio[1]), nn.Tanh(), nn.Dropout(0.2, )))
            lzzonefc.append(nn.Sequential(nn.Linear(nowio[0], nowio[1]), nn.Tanh()))
            
        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        if lzz_flag == 0:
            self.seg_layers = nn.ModuleList(seg_layers)
        if lzz_flag == 1:
            self.lzz_layers = nn.ModuleList(lzz_layers)
            self.lzz_fc = nn.ModuleList(lzz_fc)
            # self.lzz_fc = nn.ModuleList(lzzonefc)
            # self.lzz_classhead= nn.ModuleList(lzz_classhead)

    def lzzs(self):
        for index,i in enumerate(self.lzz_fc):
            nowlist = []
            for j in i:
                nowlist.extend(list(j))
            self.lzz_fc[index] = nn.Sequential(*nowlist)

        for index,i in enumerate(self.lzz_layers):
            nowlist = []
            for j in i:
                nowlist.extend([j])
            self.lzz_layers[index] = nn.Sequential(*nowlist)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        # 还是采用多层
        seg_outputs = []
        lres_input = []
        lres_input.append(skips[-1])
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input[s])
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            lres_input.append(x)
            bs, ch, num, w, h = x.shape
            # x = x.view(bs, num, ch, w, h)
            # 交换 channels 和 depth 维度
            x = x.transpose(1, 2)  # 只交换第二和第三个维度
            # x = self.lzz_layers[s](x)
            for i in range(len(self.lzz_layers[s])):
                x = self.lzz_layers[s][i](x)
            # print(x.shape)
            x_fc = x.view(bs, num, -1)
            # 有效的一版
            for i in range(len(self.lzz_fc[s])):
                x_fc = self.lzz_fc[s][i](x_fc)
                # for j in range(len(self.lzz_fc[s][i])):
                #     x_fc = self.lzz_fc[s][i][j](x_fc)
            # x_fc = self.lzz_fc[s](x_fc)
            seg_outputs.append(x_fc)
                    
        return seg_outputs
        
    def forward_org(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        classhead_flag = 0
        global lzz_flag
        # 还是采用多层
        seg_outputs = []
        lres_input = []
        lres_input.append(skips[-1])
        class_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input[s])
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            lres_input.append(x)
            if lzz_flag == 1:
                bs, ch, num, w, h = x.shape
                x = x.view(bs, num, ch, w, h)
                for i in range(len(self.lzz_layers[s])):
                    x = self.lzz_layers[s][i](x)
                # print(x.shape)
                x_fc = x.view(bs, num, -1)
                # 有效的一版
                for i in range(len(self.lzz_fc[s])):
                    # 添加分类头 只用最后一层
                    if i == len(self.lzz_fc[s]) - 1 and classhead_flag == 1:
                        x_class = self.lzz_classhead[s][0][0](x_fc)
                        for j in range(1, len(self.lzz_classhead[s][0])):
                            x_class = self.lzz_classhead[s][0][j](x_class)
                        class_outputs.append(x_class)
                    for j in range(len(self.lzz_fc[s][i])):
                        x_fc = self.lzz_fc[s][i][j](x_fc)
                
                # # 公用一层
                # for i in range(len(self.lzz_fc)):
                #     for j in range(len(self.lzz_fc[i])):
                #         x = self.lzz_fc[i][j](x)
                seg_outputs.append(x_fc)
                # print(s, "  len(torch.where(x != 0)[0]):   ", len(torch.where(x != 0)[0]))

                # # 添加分类头
                # if classhead_flag == 1:
                #     x_class = x.view(bs, num, -1)
                #     # 有效的一版
                #     for i in range(len(self.lzz_classhead[s])):
                #         for j in range(len(self.lzz_classhead[s][i])):
                #             x_class = self.lzz_classhead[s][i][j](x_class)
                #     class_outputs.append(x_class)
                
            else:
                if self.deep_supervision:
                    seg_outputs.append(self.seg_layers[s](x))
                elif s == (len(self.stages) - 1):
                    seg_outputs.append(self.seg_layers[-1](x))
        if classhead_flag == 1:
            return seg_outputs, class_outputs
        else:
            return seg_outputs
        # 新的 decoder全部改掉
        lres_input = skips[-1]
        seg_outputs = []
        x = lres_input
        bs, ch, num, w, h = x.shape
        # x = x.view(bs, num, ch, w, h)
        # for i in range(len(self.lzz_layers[0])):
        #     x = self.lzz_layers[0][i](x)
        x = x.view(bs, num, -1)
        # for i in range(len(self.lzz_fc[0])):
        #     for j in range(len(self.lzz_fc[0][i])):
        #         x = self.lzz_fc[0][i][j](x)
        for i in range(len(self.lzz_fc)):
            for j in range(len(self.lzz_fc[i])):
                x = self.lzz_fc[i][j](x)
        seg_outputs.append(x)
        return seg_outputs
    
        # 原来每层都用
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            lres_input = x
            if lzz_flag == 1:
                bs, ch, num, w, h = x.shape
                x = x.view(bs, num, ch, w, h)
                for i in range(len(self.lzz_layers[s])):
                    x = self.lzz_layers[s][i](x)
                x = x.view(bs, num, -1)
                for i in range(len(self.lzz_fc[s])):
                    for j in range(len(self.lzz_fc[s][i])):
                        x = self.lzz_fc[s][i][j](x)
                seg_outputs.append(x)
                # print(s, "  len(torch.where(x != 0)[0]):   ", len(torch.where(x != 0)[0]))
            else:
                if self.deep_supervision:
                    seg_outputs.append(self.seg_layers[s](x))
                elif s == (len(self.stages) - 1):
                    seg_outputs.append(self.seg_layers[-1](x))

        if lzz_flag == 1:
            return seg_outputs
        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]
        
        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output

class UMambaEnc(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualMambaEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        # 计算光流
        global lzz_flag
        flowflag = 0
        # if lzz_flag == 1 and flowflag == 1:
        #     flowx = x.permute(0,2,1,3,4)
        #     # 目标尺寸
        #     target_sizes = [192, 96, 48, 24, 12, 6]
        #     video_frames_tensor = flowx * 255
        #     # video_frames_tensor = video_frames_tensor.type(torch.uint8)  # 模拟真实情况，转为uint8类型
        #     # 确保Tensor在CPU上
        #     video_frames_tensor_cpu = video_frames_tensor.cpu()
        #     # 执行并行计算
        #     flows_resized = parallel_flow_computation(video_frames_tensor_cpu, target_sizes, orgflag=1)
        #     tarsize = x.shape[-1]
        #     batch = x.shape[0]
        #     empty_flow_image = np.zeros((batch, 1, tarsize, tarsize, 2))
        #     flows = np.concatenate((np.array(flows_resized), empty_flow_image), axis=1)
        #     flows = torch.from_numpy(flows).to(dtype=x.dtype, device=x.device).permute(0,4,1,2,3)
        #     x = torch.cat((flows, x), dim=1)
        #     # # 创建ProcessPoolExecutor实例来管理并行任务
        #     # with ProcessPoolExecutor() as executor:
        #     #     # 提交CPU计算任务
        #     #     future_flows = executor.submit(parallel_flow_computation, video_frames_tensor_cpu, target_sizes)
        #     #     # 在GPU上运行encoder
        #     #     skips = self.encoder(x)
        #     #     # 等待CPU上的光流计算完成
        #     #     flows_resized = future_flows.result()
        #     #     return self.decoder(skips)

        #     # 打印结果，验证每个尺寸的光流图数量
        #     # for size, flows in flows_resized.items():
        #     #     print(f"Size {size}: {len(flows)} flows")
        #     # # 假设`flows_resized`是包含192x192光流结果的列表，每个元素的shape为[192, 192, 2]
        #     # savepath = "/root/lzz2/cardiac_cycle/mamba/try_umamba/U-Mamba/data/saveflowcolor/"
            
        #     # # 假设`video_frames_tensor`是原始帧数据，形状为[48, 3, 192, 192]
        #     # for i, flow in enumerate(flows_resized[192]):
        #     #     # 调用函数转换光流为彩色图像
        #     #     color_flow_image = flow_to_color(flow)

        #     #     # 保存彩色光流图像
        #     #     cv2.imwrite(savepath + f"flow_vis_{i+1}.png", color_flow_image)
        #         # # 提取第i帧，调整维度顺序以适应OpenCV
        #         # # video_frames_tensor 的形状是 [1, 帧数, 通道, 高度, 宽度]
        #         # frame = video_frames_tensor[0, i]  # 提取第i帧，形状为 [3, 192, 192]
                
        #         # # 将Tensor转换为NumPy数组，并调整维度顺序
        #         # frame_np = frame.cpu().numpy()  # 转换为NumPy数组
        #         # frame_np = np.transpose(frame_np, (1, 2, 0))  # 调整维度顺序为 [192, 192, 3]
                
        #         # # 现在可以安全地将RGB图像转换为灰度图像
        #         # frame_gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
                
        #         # # 绘制光流并保存可视化结果，假设 draw_flow 函数已经定义
        #         # flow_vis = draw_flow(frame_gray, flow)
                
        #         # # 保存图像
        #         # cv2.imwrite(savepath + f"flow_vis_{i+1}.png", flow_vis)

        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


def get_umamba_enc_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'UMambaEnc'
    network_class = UMambaEnc
    kwargs = {
        'UMambaEnc': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    if network_class == UMambaEnc:
        model.apply(init_last_bn_before_add_to_0)

    return model
