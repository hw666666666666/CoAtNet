# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""evaluate CoAtNet on ImageNet"""


import os
import sys

import mindspore as ms
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.dataset import create_dataset
from src.utils import context_device_init, do_keep_cell_fp32
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper, modelarts_process
from src.model_utils.device_adapter import get_device_id
from src.coatnet import CoAtNet, coatnet_0, coatnet_1, coatnet_2


def process_checkpoint(net, ckpt):
    prefix = "ema."
    len_prefix = len(prefix)
    if config.enable_ema:
        ema_ckpt = {}
        for name, param in ckpt.items():
            if name.startswith(prefix):
                ema_ckpt[name[len_prefix:]] = ms.Parameter(default_input=param.data, name=param.name[len_prefix:])
        ckpt = ema_ckpt

    net_param_dict = net.parameters_dict()
    ckpt = {k:v for k, v in ckpt.items() if k in net_param_dict}

    for name, param in ckpt.items():
        if "relative_position_bias_table" in name:
            size = net_param_dict[name].data.shape[1:]
            ckpt_param_data = ms.ops.ExpandDims()(param.data, 0)
            ckpt_param_data = ms.ops.ResizeBilinear(size, True)(ckpt_param_data)
            ckpt_param_data = ms.ops.Squeeze(0)(ckpt_param_data)
            ckpt[name] = ms.Parameter(default_input=ckpt_param_data, name=name)

    return ckpt


@moxing_wrapper(pre_process=modelarts_process)
def eval_mobilenetv2():
    config.batch_size = 100
    config.dataset_path = os.path.join(config.dataset_path, 'val')

    # When test the pretrained model on 224 * 224 resolution
    # config.center_crop = True
    # config.enable_ema = False

    # When test the finetuned model on 384 * 384 resolution
    # config.center_crop = False
    # config.enable_ema = True
    # config.image_height = 384
    # config.image_width = 384

    if not config.device_id:
        config.device_id = get_device_id()
    context_device_init(config)
    print('\nconfig: {} \n'.format(config))

    # CoAtNet-0
    # net = CoAtNet((config.image_height, config.image_width), 3, [2, 3, 5, 2], [64, 96, 192, 384, 768], drop_path_rate=0.2, num_classes=config.num_classes)
    # CoAtNet-1
    # net = CoAtNet((config.image_height, config.image_width), 3, [2, 6, 14, 2], [64, 96, 192, 384, 768], drop_path_rate=0.3, num_classes=config.num_classes)
    # CoAtNet-2
    # net = CoAtNet((config.image_height, config.image_width), 3, [2, 6, 14, 2], [128, 128, 256, 512, 1024], drop_path_rate=0.5, num_classes=config.num_classes)
    net = getattr(sys.modules[__name__], config.architecture)(config.image_height, config.image_width)

    ckpt = load_checkpoint(config.load_path)

    ckpt = process_checkpoint(net, ckpt)

    load_param_into_net(net, ckpt)

    net.to_float(ms.dtype.float16)
    # do_keep_cell_fp32(net, cell_types=(nn.Softmax, nn.BatchNorm2d, nn.LayerNorm, nn.GELU))
    # do_keep_cell_fp32(net, cell_types=(nn.Softmax, nn.BatchNorm2d, nn.LayerNorm))
    do_keep_cell_fp32(net, cell_types=(nn.BatchNorm2d, nn.LayerNorm, nn.Softmax, nn.GELU, nn.Tanh, nn.Sigmoid))

    dataset = create_dataset(dataset_path=config.dataset_path, do_train=False, config=config, drop_remainder=False)
    step_size = dataset.get_dataset_size()
    if step_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images count of eval dataset is more \
            than batch_size in config.py")
    print("step_size = ", step_size)
    net.set_train(False)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    metrics = {'Validation-Loss': nn.Loss(),
               'Top1-Acc': nn.Top1CategoricalAccuracy(),
               'Top5-Acc': nn.Top5CategoricalAccuracy()}
    model = ms.Model(net, loss_fn=loss, metrics=metrics)

    res = model.eval(dataset)
    print("result:{}\npretrain_ckpt={}".format(res, config.load_path))


if __name__ == '__main__':
    eval_mobilenetv2()
