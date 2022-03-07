from torch.utils.tensorboard import SummaryWriter
from base_config import BaseConfigByEpoch
from model_map import get_model_fn
from data.data_factory import create_dataset, load_cuda_data, num_iters_per_epoch
from torch.nn.modules.loss import CrossEntropyLoss
from utils.pyt_utils import ensure_dir
from utils.misc import torch_accuracy, AvgMeter
from collections import OrderedDict
from tqdm import tqdm
import time
from utils.lr_scheduler import get_lr_scheduler
import os
from ndp_test import val_during_train
from utils.checkpoint import get_last_checkpoint
from rr.resrep_util import *
from rr.resrep_convert import *
from utils.engine import Engine

TRAIN_SPEED_START = 0.1
TRAIN_SPEED_END = 0.2

COLLECT_TRAIN_LOSS_EPOCHS = 3

TEST_BATCH_SIZE = 100

CONVERSION_EPSILON = 1e-5

def train_one_step(resrep_config:ResRepConfig,
                   net, data, label, optimizer, criterion, if_accum_grad = False,
                   cur_flops=None):
    pred = net(data)
    # cur_flops=np.log(cur_flops)
    # if cur_flops is not None:
    #     loss = criterion(pred, label)*cur_flops
    # else:
    loss=criterion(pred, label)
    loss.backward()


    optimizer.step()
    optimizer.zero_grad()
    acc, acc5 = torch_accuracy(pred, label, (1,5))
    return acc, acc5, loss

def sgd_optimizer(cfg, resrep_config:ResRepConfig, model, no_l2_keywords, use_nesterov, keyword_to_lr_mult):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.base_lr
        weight_decay = cfg.weight_decay
        if "bias" in key or "bn" in key or "BN" in key:
            weight_decay = cfg.weight_decay_bias
            print('set weight_decay_bias={} for {}'.format(weight_decay, key))
        for kw in no_l2_keywords:
            if kw in key:
                weight_decay = 0
                print('NOTICE! weight decay = 0 for ', key, 'because {} in {}'.format(kw, key))
                break
        if 'bias' in key:
            apply_lr = 2 * lr
        else:
            apply_lr = lr
        if keyword_to_lr_mult is not None:
            for keyword, mult in keyword_to_lr_mult.items():
                if keyword in key:
                    apply_lr *= mult
                    print('multiply lr of {} by {}'.format(key, mult))
                    break
        if 'compactor' in key:
            use_momentum = resrep_config.compactor_momentum
            print('momentum {} for {}'.format(use_momentum, key))
        else:
            use_momentum = cfg.momentum
        params += [{"params": [value], "lr": apply_lr, "weight_decay": weight_decay, "momentum": use_momentum}]
    optimizer = torch.optim.SGD(params, lr, momentum=cfg.momentum, nesterov=use_nesterov)
    return optimizer



def get_optimizer(cfg, resrep_config, model, no_l2_keywords, use_nesterov=False, keyword_to_lr_mult=None):
    return sgd_optimizer(cfg, resrep_config, model, no_l2_keywords, use_nesterov=use_nesterov, keyword_to_lr_mult=keyword_to_lr_mult)

def get_criterion(cfg):
    return CrossEntropyLoss()
def covariance(vec1):
    X=vec1.detach().numpy()
    return np.cov(X)
def similarity_kl(metric1,metric2):
    metric1=torch.Tensor(metric1)
    metric2=torch.Tensor(metric2)
    metric2_1 = torch.inverse(metric2)
    out_kl = 1/2*(torch.trace(torch.matmul(metric2_1,metric1))-torch.log(torch.norm(metric1,p=1)/torch.norm(metric2,p=1)))
    return out_kl
def similarity(metric1,metric2):
    metric1=torch.Tensor(metric1)
    metric2=torch.Tensor(metric2)
    tr = torch.trace(torch.matmul(metric1,metric2))
    out = tr/(torch.norm(metric1)*torch.norm(metric2))
    return out
def compute_similarity(model):
    dict2compa = {}
    namelist = []
    paramlist = []
    similarlist = []
    for name, param in model.named_parameters():
        if 'conv.weight' in name:
            dict2compa[name] = param
            paramlist.append(param)
            namelist.append(name)
            print(param.size())
    for i in range(len(paramlist)):
        metric1 = paramlist[i].view(paramlist[i].size()[0], -1).cpu()
        co_metric1 = covariance(metric1)
        for j in range(i+1,len(paramlist)):
            if paramlist[i].size() != paramlist[j].size():
                continue
            metric2 = paramlist[j].view(paramlist[j].size()[0], -1).cpu()
            co_metric2 = covariance(metric2)
            similar = similarity(co_metric1,co_metric2)
            similarlist.append((similar,(i,j)))
    similarlist=sorted(similarlist,key=lambda x:x[0])
    print(1)
    return similarlist
def compute_flops(model):
    from thop import profile
    inpu=torch.randn(size=(1,3,32,32)).cuda()
    flops,params = profile(model,inputs=(inpu,))
    print('flops:{}'.format(flops))
def layer_zeros(model:nn.Module,fusion_mask):
    conv_idx = -1
    for name,param in model.named_parameters():
        if 'conv' in name:
            if 'conv.weight' in name:
                conv_idx+=1
            if fusion_mask[conv_idx]==0:
                if param.grad is not None:
                    param.data = 0 * param.data
                    param.grad=0*param.grad
                    param.requires_grad=False

def resrep_train_main(
        local_rank,
        cfg:BaseConfigByEpoch, resrep_config:ResRepConfig, resrep_builder,
        net=None, train_dataloader=None, val_dataloader=None, show_variables=False,
               init_hdf5=None, no_l2_keywords='depth', gradient_mask=None, use_nesterov=False,
               load_weights_keyword=None,
               keyword_to_lr_mult=None,
               auto_continue=False,save_hdf5_epochs=5):
    #1.第一个epoch对所有的层进行相似度计算
    #2.
    if no_l2_keywords is None:
        no_l2_keywords = []
    if type(no_l2_keywords) is not list:
        no_l2_keywords = [no_l2_keywords]

    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.tb_dir)
    with Engine(local_rank=local_rank) as engine:
        engine.setup_log(
            name='train', log_dir=cfg.output_dir, file_name='log.txt')

        # ----------------------------- build model ------------------------------
        if net is None:
            net_fn = get_model_fn(cfg.dataset_name, cfg.network_type)
            model = net_fn(cfg, resrep_builder)
        else:
            model = net
        model = model.cuda()
        # ----------------------------- model done ------------------------------

        # ---------------------------- prepare data -------------------------
        if train_dataloader is None:
            train_data = create_dataset(cfg.dataset_name, cfg.dataset_subset,
                                        cfg.global_batch_size, distributed=engine.distributed)
        if cfg.val_epoch_period > 0 and val_dataloader is None:
            val_data = create_dataset(cfg.dataset_name, 'val',
                                      global_batch_size=100, distributed=False)
        engine.echo('NOTE: Data prepared')
        engine.echo('NOTE: We have global_batch_size={} on {} GPUs, the allocated GPU memory is {}'.format(cfg.global_batch_size, torch.cuda.device_count(), torch.cuda.memory_allocated()))
        # ----------------------------- data done --------------------------------

        # ------------------------ parepare optimizer, scheduler, criterion -------
        optimizer = get_optimizer(cfg, resrep_config, model,
                                  no_l2_keywords=no_l2_keywords, use_nesterov=use_nesterov, keyword_to_lr_mult=keyword_to_lr_mult)
        scheduler = get_lr_scheduler(cfg, optimizer)
        criterion = get_criterion(cfg).cuda()
        # --------------------------------- done -------------------------------

        engine.register_state(
            scheduler=scheduler, model=model, optimizer=optimizer)

        #   finish.hdf5 exists, do conversion and return
        if os.path.exists(os.path.join(cfg.output_dir, 'finish.hdf5')):
            engine.load_hdf5(os.path.join(cfg.output_dir, 'finish.hdf5'), load_weights_keyword=load_weights_keyword)
            if 'mi1' in cfg.network_type:
                compactor_convert_mi1(model=model, origin_deps=cfg.deps, thresh=CONVERSION_EPSILON,
                                      save_path=os.path.join(cfg.output_dir,
                                                             'finish_converted.hdf5'))
            else:
                compactor_convert(model=model, origin_deps=cfg.deps, thresh=CONVERSION_EPSILON,
                                  pacesetter_dict=resrep_config.pacesetter_dict,
                                  succ_strategy=resrep_config.succeeding_strategy,
                                  save_path=os.path.join(cfg.output_dir, 'finish_converted.hdf5'))

            return

        if engine.distributed:
            torch.cuda.set_device(local_rank)
            engine.echo('Distributed training, device {}'.format(local_rank))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank],
                broadcast_buffers=False, )
        else:
            assert torch.cuda.device_count() == 1
            engine.echo('Single GPU training')

        if cfg.init_weights:
            engine.load_checkpoint(cfg.init_weights)
        if init_hdf5:
            engine.load_hdf5(init_hdf5, load_weights_keyword=load_weights_keyword)

        if auto_continue:
            assert cfg.init_weights is None
            engine.load_checkpoint(get_last_checkpoint(cfg.output_dir))
        if show_variables:
            engine.show_variables()

        # ------------ do training ---------------------------- #
        engine.log("\n\nStart training with pytorch version {}".format(torch.__version__))

        iteration = engine.state.iteration
        iters_per_epoch = num_iters_per_epoch(cfg)
        max_iters = iters_per_epoch * cfg.max_epochs
        tb_writer = SummaryWriter(cfg.tb_dir)
        tb_tags = ['Top1-Acc', 'Top5-Acc', 'Loss']

        model.train()

        done_epochs = iteration // iters_per_epoch
        last_epoch_done_iters = iteration % iters_per_epoch

        if done_epochs == 0 and last_epoch_done_iters == 0:
            engine.save_hdf5(os.path.join(cfg.output_dir, 'init.hdf5'))

        recorded_train_time = 0
        recorded_train_examples = 0

        collected_train_loss_sum = 0
        collected_train_loss_count = 0

        similarity_list = compute_similarity(model)
        compute_flops(model)
        layer_fusion_per_epochs=5
        fusion_mask = np.random.randint(2,size=58)
        mask_times=0
        for epoch in range(done_epochs, cfg.max_epochs):

            # if epoch == 0 and engine.local_rank == 0:
                # val_during_train(epoch=epoch, iteration=iteration, tb_tags=tb_tags, engine=engine, model=model,
                #              val_data=val_data, criterion=criterion,
                #                  descrip_str='Begin',
                #              dataset_name=cfg.dataset_name, test_batch_size=TEST_BATCH_SIZE, tb_writer=tb_writer)
            if (epoch+1) % layer_fusion_per_epochs ==0:
                mask_times+=1
                for i in range(mask_times):
                    for j in range(len(fusion_mask)):
                        if fusion_mask[j]==0:
                            fusion_mask[j]=1
                            break
                layer_zeros(model,fusion_mask=fusion_mask)

            if engine.distributed and hasattr(train_data, 'train_sampler'):
                train_data.train_sampler.set_epoch(epoch)

            if epoch == done_epochs:
                pbar = tqdm(range(iters_per_epoch - last_epoch_done_iters))
            else:
                pbar = tqdm(range(iters_per_epoch))

            top1 = AvgMeter()
            top5 = AvgMeter()
            losses = AvgMeter()
            discrip_str = 'Epoch-{}/{}'.format(epoch, cfg.max_epochs)
            pbar.set_description('Train' + discrip_str)

            for _ in pbar:

                start_time = time.time()
                data, label = load_cuda_data(train_data, dataset_name=cfg.dataset_name)

                    # load_cuda_data(train_dataloader, cfg.dataset_name)
                data_time = time.time() - start_time

                if_accum_grad = ((iteration % cfg.grad_accum_iters) != 0)

                train_net_time_start = time.time()

                acc, acc5, loss = train_one_step(resrep_config, model, data, label, optimizer,
                                                 criterion,
                                                 if_accum_grad)
                train_net_time_end = time.time()

                if iteration > TRAIN_SPEED_START * max_iters and iteration < TRAIN_SPEED_END * max_iters:
                    recorded_train_examples += cfg.global_batch_size
                    recorded_train_time += train_net_time_end - train_net_time_start

                scheduler.step()

                for module in model.modules():
                    if hasattr(module, 'set_cur_iter'):
                        module.set_cur_iter(iteration)

                if iteration % cfg.tb_iter_period == 0 and engine.world_rank == 0:
                    for tag, value in zip(tb_tags, [acc.item(), acc5.item(), loss.item()]):
                        tb_writer.add_scalars(tag, {'Train': value}, iteration)

                top1.update(acc.item())
                top5.update(acc5.item())
                losses.update(loss.item())

                if epoch >= cfg.max_epochs - COLLECT_TRAIN_LOSS_EPOCHS:
                    collected_train_loss_sum += loss.item()
                    collected_train_loss_count += 1

                pbar_dic = OrderedDict()
                pbar_dic['data-time'] = '{:.2f}'.format(data_time)
                pbar_dic['cur_iter'] = iteration
                pbar_dic['lr'] = scheduler.get_lr()[0]
                pbar_dic['top1'] = '{:.5f}'.format(top1.mean)
                pbar_dic['top5'] = '{:.5f}'.format(top5.mean)
                pbar_dic['loss'] = '{:.5f}'.format(losses.mean)
                pbar.set_postfix(pbar_dic)

                iteration += 1

                if iteration >= max_iters or iteration % cfg.ckpt_iter_period == 0:
                    engine.update_iteration(iteration)
                    if (not engine.distributed) or (engine.distributed and engine.world_rank == 0):
                        engine.save_and_link_checkpoint(cfg.output_dir)
                if iteration >= max_iters:
                    break

            engine.update_iteration(iteration)
            engine.save_latest_ckpt(cfg.output_dir)

            if (epoch + 1) % save_hdf5_epochs == 0:
                engine.save_hdf5(os.path.join(cfg.output_dir, 'epoch-{}.hdf5'.format(epoch)))

            if local_rank == 0 and \
                    cfg.val_epoch_period > 0 and (epoch >= cfg.max_epochs - 10 or epoch % cfg.val_epoch_period == 0):
                val_during_train(epoch=epoch, iteration=iteration, tb_tags=tb_tags, engine=engine, model=model,
                                 val_data=val_data, criterion=criterion, descrip_str=discrip_str,
                                 dataset_name=cfg.dataset_name, test_batch_size=TEST_BATCH_SIZE, tb_writer=tb_writer)

            if iteration >= max_iters:
                break

        #   do something after the training
        if recorded_train_time > 0:
            exp_per_sec = recorded_train_examples / recorded_train_time
        else:
            exp_per_sec = 0
        engine.log(
            'TRAIN speed: from {} to {} iterations, batch_size={}, examples={}, total_net_time={:.4f}, examples/sec={}'
            .format(int(TRAIN_SPEED_START * max_iters), int(TRAIN_SPEED_END * max_iters), cfg.global_batch_size,
                    recorded_train_examples, recorded_train_time, exp_per_sec))
        if cfg.save_weights:
            engine.save_checkpoint(cfg.save_weights)
            print('NOTE: training finished, saved to {}'.format(cfg.save_weights))
        engine.save_hdf5(os.path.join(cfg.output_dir, 'finish.hdf5'))
        if collected_train_loss_count > 0:
            engine.log('TRAIN LOSS collected over last {} epochs: {:.6f}'.format(COLLECT_TRAIN_LOSS_EPOCHS,
                                                                                 collected_train_loss_sum / collected_train_loss_count))
        if 'mi1' in cfg.network_type:
            compactor_convert_mi1(model=model, origin_deps=cfg.deps, thresh=CONVERSION_EPSILON,
                                  save_path=os.path.join(cfg.output_dir, 'finish_converted.hdf5'))
        else:
            compactor_convert(model=model, origin_deps=cfg.deps, thresh=CONVERSION_EPSILON,
                              pacesetter_dict=resrep_config.pacesetter_dict,
                              succ_strategy=resrep_config.succeeding_strategy,
                              save_path=os.path.join(cfg.output_dir, 'finish_converted.hdf5'))


