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
"""Train CoAtNet on ImageNet"""


import os
import sys
import time

import mindspore as ms
import mindspore.nn as nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from src.dataset import create_dataset
from src.utils import get_lr, do_keep_cell_fp32, build_params_groups, context_device_init, count_params
from src.metrics import DistAccuracy, ClassifyCorrectCell
from src.callbacks import Monitor
from src.cell_wrapper import CustomTrainOneStepWithLossScaleCell
from src.model_utils.moxing_adapter import moxing_wrapper, modelarts_process
from src.model_utils.device_adapter import get_device_id
from src.model_utils.config import config
from src.coatnet import CoAtNet, coatnet_0, coatnet_1, coatnet_2


@moxing_wrapper(pre_process=modelarts_process)
def train():
    config.train_dataset_path = os.path.join(config.dataset_path, 'train')
    config.eval_dataset_path = os.path.join(config.dataset_path, 'val')

    if not config.device_id:
        config.device_id = get_device_id()
    start = time.time()
    # set context and device init
    context_device_init(config)
    print('\nconfig: {} \n'.format(config))

    dataset = create_dataset(dataset_path=config.train_dataset_path, do_train=True, config=config,
                             enable_cache=config.enable_cache, cache_session_id=config.cache_session_id)
    step_size = dataset.get_dataset_size()

    if step_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images' count of train dataset is more \
            than batch_size in config.py")

    # get learning rate
    lr = ms.Tensor(get_lr(global_step=0,
                          lr_max=config.lr_max,
                          lr_min=config.lr_min,
                          lr_warmup_epochs=config.lr_warmup_epochs,
                          total_epochs=config.num_epochs,
                          steps_per_epoch=step_size))

    metrics = None
    dist_eval_network = None
    eval_dataset = None

    # define network
    # CoAtNet-0
    # net = CoAtNet((config.image_height, config.image_width), 3, [2, 3, 5, 2], [64, 96, 192, 384, 768], drop_path_rate=0.2, num_classes=config.num_classes)
    # CoAtNet-1
    # net = CoAtNet((config.image_height, config.image_width), 3, [2, 6, 14, 2], [64, 96, 192, 384, 768], drop_path_rate=0.3, num_classes=config.num_classes)
    # CoAtNet-2
    # net = CoAtNet((config.image_height, config.image_width), 3, [2, 6, 14, 2], [128, 128, 256, 512, 1024], drop_path_rate=0.5, num_classes=config.num_classes)
    net = getattr(sys.modules[__name__], config.architecture)(config.image_height, config.image_width)

    if config.rank_id == 0:
        print("Total number of parameters: {}".format(count_params(net)))

    # mixed precision training
    net.to_float(ms.dtype.float16)
    do_keep_cell_fp32(net, cell_types=(nn.BatchNorm2d, nn.LayerNorm, nn.Softmax, nn.GELU, nn.Tanh, nn.Sigmoid))

    if config.run_eval:
        metrics = {'acc': DistAccuracy(batch_size=config.batch_size, device_num=config.rank_size)}
        dist_eval_network = ClassifyCorrectCell(net, config.run_distribute)
        eval_dataset = create_dataset(dataset_path=config.eval_dataset_path, do_train=False, config=config)

    group_params = build_params_groups(net, config.weight_decay)

    # define loss
    # label smoothing and mixup are done with dataset pipeline
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
    # mixed precision training
    net_with_loss = ms.amp._add_loss_network(net, loss, ms.dtype.float16)

    opt = nn.AdamWeightDecay(params=group_params, learning_rate=lr, eps=config.epsilon)

    if config.device_target in ["Ascend", "GPU"]:
        scale_sense = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 24,
                                                                    scale_factor=2, scale_window=2000)
        train_net = CustomTrainOneStepWithLossScaleCell(net_with_loss, opt, scale_sense,
                                                        config.enable_ema, config.ema_decay,
                                                        config.enable_clip_norm, config.gradient_norm)
    else:
        raise ValueError

    model = ms.Model(train_net, metrics=metrics, eval_network=dist_eval_network)

    # add callbacks
    cb = [Monitor(lr_init=lr.asnumpy(), model=model, eval_dataset=eval_dataset)]

    ckpt_prefix = "coatnet"
    ckpt_save_dir = os.path.join(config.save_checkpoint_path, "ckpt_" + str(config.rank_id))
    if config.save_checkpoint and config.rank_id == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix=ckpt_prefix, directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]

    print("============== Starting Training ==============")
    model.train(config.num_epochs, dataset, callbacks=cb, dataset_sink_mode=True)
    print("============== End Training ==============")

    if config.enable_cache:
        print("Remember to shut down the cache server via \"cache_admin --stop\"")


if __name__ == '__main__':
    ms.set_seed(1)
    train()
