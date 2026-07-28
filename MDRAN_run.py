import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from src.utils import loss,prompt_tuning,IID_losses
from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import  ImageList_idx, ImageList_idx_aug_fix
from sklearn.metrics import confusion_matrix
from clip.custom_clip import get_coop
import clip
import pandas as pd
from src.utils.utils import *
logger = logging.getLogger(__name__)
from collections import Counter




def select_best_domain(tensor,median, vars):
    """
    根据每个样本的最大分类置信度选择最佳域（domain）

    参数:
        tensor: 形状为[domain, batch, class]的3D张量，
                class维度是softmax输出的概率分布

    返回:
        最佳[batch, class]张量
    """
    # 标准化
    median_t = torch.tensor(median, device=tensor.device).view(-1, 1, 1)
    vars_t = torch.tensor(vars, device=tensor.device).view(-1, 1, 1)

    tensor = (tensor - median_t) / vars_t

    tensor = torch.abs(tensor)

    confidence, _ = torch.max(tensor, dim=2)  # 形状: [domain, batch]


    # 找到每个样本具有最高置信度的域索引
    best_domain_indices = torch.argmax(confidence, dim=0)  # 形状: [batch]

    # 创建batch维度的索引 [0, 1, 2, ..., batch_size-1]
    batch_indices = torch.arange(tensor.size(1), device=tensor.device)

    # 选择最佳域的class向量
    best_output = tensor[best_domain_indices, batch_indices, :]  # 形状: [batch, class]

    return best_output

def data_load(cfg): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.TEST.BATCH_SIZE
    txt_tar = open(cfg.t_dset_path).readlines()
    txt_test = open(cfg.test_dset_path).readlines()
    if not cfg.DA == 'uda':
        label_map_s = {}
        for i in range(len(cfg.src_classes)):
            label_map_s[cfg.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in cfg.tar_classes:
                if int(reci[1]) in cfg.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()
    dsets["target"] = ImageList_idx_aug_fix(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["source"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=2, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=True)
    return dset_loaders

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  #else:
    #normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])
def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  #else:
   # normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    median = [np.nanmedian(t.cpu().numpy()) for t in all_output]
    vars = [t.var().item() for t in all_label]
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent, median, vars


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def test_time_tuning(model, inputs, optimizer, cfg, target_output):
    target_output = target_output.cuda()
    target_output =  nn.Softmax(dim=1)(target_output)
    for j in range(cfg.ProDe.TTA_STEPS):
        with torch.amp.autocast('cuda'):
            output_logits,_ = model(inputs)
            # show_inputs = np.array(inputs.cpu())
            output = nn.Softmax(dim=1)(output_logits)
            iic_loss = IID_losses.IID_loss(output, target_output)
        optimizer.zero_grad()
        iic_loss.backward()
        optimizer.step()
    return output

def test_time_adapt_eval(input, model, optimizer, optim_state, cfg, target_output, flg = True):
    optimizer.load_state_dict(optim_state)
    if flg == True:
        output = test_time_tuning(model, input, optimizer, cfg, target_output)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            model.eval()
            output,_ = model(input)
    return output

def train_target(cfg):
    text_inputs = clip_pre_text(cfg)
    dset_loaders = data_load(cfg)
    model=list()
    for i in range(len(cfg.domain) - 1):
        model.append(get_coop(cfg.ProDe.ARCH, cfg.SETTING.DATASET, int(cfg.GPU_ID), cfg.ProDe.N_CTX, cfg.ProDe.CTX_INIT))
    for i in range(len(cfg.domain) - 1):
        for name, param in model[i].named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)


    ## set base network
    netF = list()
    for i in range(len(cfg.domain)-1):
        if cfg.MODEL.ARCH[0:3] == 'res':
            netF.append(network.ResBase(res_name=cfg.MODEL.ARCH).cuda())
        elif cfg.MODEL.ARCH[0:3] == 'vgg':
            netF.append(network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda())

    netB = list()
    netC = list()
    for i in range(len(cfg.domain)-1):
        netB.append(network.feat_bottleneck(type='bn', feature_dim=netF[i].in_features, bottleneck_dim=cfg.bottleneck).cuda())
        netC.append(network.feat_classifier(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda())

    for i in range(len(cfg.domain)-1):
        modelpath = cfg.output_dir_src_multi[i] + '/source_F.pt'
        netF[i].load_state_dict(torch.load(modelpath))
        modelpath = cfg.output_dir_src_multi[i] + '/source_B.pt'
        netB[i].load_state_dict(torch.load(modelpath))
        modelpath = cfg.output_dir_src_multi[i] + '/source_C.pt'
        netC[i].load_state_dict(torch.load(modelpath))

    param_group = list()
    for i in range(len(cfg.domain)-1):
        param_group.append([])
    for i in range(len(cfg.domain) - 1):
        for k, v in netF[i].named_parameters():
            if cfg.OPTIM.LR_DECAY1 > 0:
                param_group[i] += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]
            else:
                v.requires_grad = False
        for k, v in netB[i].named_parameters():
            if cfg.OPTIM.LR_DECAY2 > 0:
                param_group[i] += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY2}]
            else:
                v.requires_grad = False
        for k, v in netC[i].named_parameters():
            if cfg.OPTIM.LR_DECAY1 > 0:
                param_group[i] += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]
            else:
                v.requires_grad = False


    param_group_ib = list()
    for i in range(len(cfg.domain) - 1):
        param_group_ib_single=[]
        for k, v in model[i].prompt_learner.named_parameters():
            if(v.requires_grad == True):
                param_group_ib_single += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]
        param_group_ib.append(param_group_ib_single)

    optimizer=list()
    for i in range(len(cfg.domain) - 1):
        optimizer_single = optim.SGD(param_group[i])
        optimizer_single = op_copy(optimizer_single)
        optimizer.append(optimizer_single)

    optimizer_ib = list()
    optim_state= list()
    for i in range(len(cfg.domain) - 1):
        optimizer_ib_single = optim.SGD(param_group_ib[i])
        optimizer_ib_single = op_copy(optimizer_ib_single)
        optim_state_single = deepcopy(optimizer_ib_single.state_dict())
        optim_state.append(optim_state_single)
        optimizer_ib.append(optimizer_ib_single)
    for i in range(len(cfg.domain) - 1):
        model[i].reset_classnames(cfg.classname, cfg.ProDe.ARCH)
    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0
    num_sample=len(dset_loaders["target"].dataset)
    loader = dset_loaders["source"]

    logtis_bank = list()
    for i in range(len(cfg.domain) - 1):
        logtis_bank.append(torch.randn(num_sample, cfg.class_num).cuda())

    label_bank = torch.ones(num_sample).cuda()
    clip_bank = torch.randn(num_sample, cfg.class_num).cuda()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            label = data[1].cuda()
            indx=data[-1]
            inputs = inputs.cuda()
            label_bank[indx] = label.float()
            for i in range(len(cfg.domain) - 1):
                output = netB[i](netF[i](inputs))
                outputs = netC[i](output)
                logtis_bank[i][indx] = outputs.detach().clone()
            clip_score = test_time_adapt_eval(inputs, model[0], optimizer_ib[0], optim_state[0], cfg,
                                              outputs,  flg = False)
            clip_bank[indx] = clip_score
    median = [np.nanmedian(t.cpu().numpy()) for t in logtis_bank]
    vars = [t.var().item() for t in logtis_bank]
    means = [t.mean().item() for t in logtis_bank]
    vars_clip = [t.var().item() for t in clip_bank]
    means_clip = [t.mean().item() for t in clip_bank]


    while iter_num < max_iter:


        try:
            (inputs_test, inputs_test_augs), _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            (inputs_test, inputs_test_augs), _, tar_idx = next(iter_test)
        if inputs_test.size(0) == 1:
            continue
        if iter_num % interval_iter == 0:
            domain_acc_list = []
            combined_acc_list = []
        inputs_test = inputs_test.cuda()
        inputs_test_augs = inputs_test_augs[0].cuda()

        iter_num += 1


        softmax_out=list()
        outputs_test_new=list()
        outputs_test_list = list()
        for i in range(len(cfg.domain) - 1):
            lr_scheduler(optimizer[i], iter_num=iter_num, max_iter=max_iter)
            features_test = netB[i](netF[i](inputs_test))
            outputs_test = netC[i](features_test)
            outputs_test_list.append(outputs_test)
            softmax_out.append(nn.Softmax(dim=1)(outputs_test))
            with torch.no_grad():
                outputs_test_new.append(outputs_test.clone().detach())

        if cfg.ProDe.ARCH == 'RN50':
            inputs_test_clip = inputs_test_augs
        else:
            inputs_test_clip = inputs_test

        outputs_test_new = torch.stack(outputs_test_new)


        #LND：

        outputs_test_new = select_best_domain(outputs_test_new, median, vars)

        means_t = torch.tensor(means, device=outputs_test_new.device, dtype=outputs_test_new.dtype).view(-1, 1, 1)
        vars_t = torch.tensor(vars, device=outputs_test_new.device, dtype=outputs_test_new.dtype).view(-1, 1, 1)

        outputs_test_new = (outputs_test_new - means_t) / (vars_t + 1e-8)


        clip_score = test_time_adapt_eval(inputs_test_clip, model[0], optimizer_ib[0], optim_state[0], cfg,
                                          outputs_test_new)
        clip_score = clip_score.float()

        means_clip_t = torch.tensor(means_clip, device=outputs_test_new.device, dtype=outputs_test_new.dtype).view(-1, 1, 1)
        vars_clip_t = torch.tensor(vars_clip, device=outputs_test_new.device, dtype=outputs_test_new.dtype).view(-1, 1, 1)

        clip_score = (clip_score - means_clip_t) / (vars_clip_t + 1e-8)


        for i in range(len(cfg.domain) - 1):

            with (torch.no_grad()):

                new_clip = outputs_test_new + clip_score.cuda()

                clip_score_sm = nn.Softmax(dim=1)(new_clip)


            _, clip_index_new = torch.max(new_clip, 1)
            clip_score_sm=(nn.Softmax(dim=1)(new_clip))

            iic_loss = IID_losses.IID_loss(softmax_out[i], clip_score_sm)
            classifier_loss = cfg.ProDe.IIC_PAR * iic_loss
            msoftmax = softmax_out[i].mean(dim=0)

            if  cfg.SETTING.DATASET=='office':
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.ProDe.EPSILON))
                classifier_loss = classifier_loss - 1.0 * gentropy_loss
            if  cfg.SETTING.DATASET=='office-home':
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.ProDe.EPSILON))
                classifier_loss = classifier_loss - 1.0 * gentropy_loss
            if  cfg.SETTING.DATASET=='VISDA-C':
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.ProDe.EPSILON))
                classifier_loss = classifier_loss - 0.1 * gentropy_loss
            if cfg.SETTING.DATASET == 'domainnet126':
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.ProDe.EPSILON))
                classifier_loss = classifier_loss - 0.01 * gentropy_loss
            pred = clip_index_new

            entropy_loss = nn.CrossEntropyLoss()(outputs_test_list[i], pred)

            classifier_loss =  cfg.ProDe.GENT_PAR*entropy_loss + classifier_loss

            optimizer[i].zero_grad()
            classifier_loss.backward()
            optimizer[i].step()



            if iter_num % interval_iter == 0 or iter_num == max_iter:
                netF[i].eval()
                netB[i].eval()
                netC[i].eval()
                print('这是第'+str(i)+'个模型的准确率')
                if cfg.SETTING.DATASET=='VISDA-C':
                    acc_s_te, acc_list, median, vars = cal_acc(dset_loaders['test'], netF[i], netB[i], netC[i], True)
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,classifier_loss) + '\n' + acc_list
                else:
                    acc_s_te, _, median, vars = cal_acc(dset_loaders['test'], netF[i], netB[i], netC[i], False)
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,classifier_loss)

                logger.info(log_str)
                netF[i].train()
                netB[i].train()
                netC[i].train()

        
    return netF, netB, netC


def print_cfg(cfg):
    s = "==========================================\n"    
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def clip_pre_text(cfg):
    List_rd = []
    with open(cfg.name_file) as f:
        for line in f:
            List_rd.extend([i for i in line.split()])
    f.close()
    classnames = List_rd
    classnames = [name.replace("_", " ") for name in classnames]
    cfg.classname = classnames
    prompt_prefix = cfg.ProDe.CTX_INIT.replace("_"," ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
    return tokenized_prompts
