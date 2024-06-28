import sys
import argparse
import multiprocessing
import h5py
import numpy as np
import os
import torch
import datetime
from tqdm import tqdm
import sys
import importlib
import shutil
import torch.nn as nn
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score
import utils.provider as provider
from utils.provider import EarlyStopping
from torch.utils.data import Dataset
import random
from pathlib import Path
from matplotlib import pyplot as plt
import logging
from tensorboardX import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

device = torch.device('cuda:1')
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def parse_args():
  '''PARAMETERS'''
  parser = argparse.ArgumentParser('SurfProNN Training')
  parser.add_argument('--batch_size', type=int, default=24, help='batch size in training [default: 24]')
  parser.add_argument('--model', default='point_pn_plm', help='model name [default: pointnet2_msg]')
  parser.add_argument('--gpu', type=str, default='3', help='specify gpu device [default: 0]')
  parser.add_argument('--num_point', type=int, default=1000, help='Point Number [default: 50]')
  parser.add_argument('--epoch', default=10, type=int, help='number of epoch in training [default: 30]')
  parser.add_argument('--learning_rate', default=1e-3, type=float,
                      help='learning rate in training [default: 0.001,finetune:1e-4]')
  parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
  parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
  parser.add_argument('--seed', type=int, default=42, help='random seed')
  parser.add_argument('--use_res', type=bool, default=True, help='Use residual features')
  parser.add_argument('--document_dir', type=str, default='dockground',
                      help='complex infos root [capri or dockground]')
  parser.add_argument('--data_dir', type=str, default='dockground_1000_dmasif',
                      help='experiment root [capri_1000,dockground_1000_dmasif]')
  parser.add_argument('--log_dir', type=str, default='dockground_1000_dmasif', help='experiment root')
  parser.add_argument('--checkpoint', type=str, default='SurfProNN/dmasif', help='checkpoints path such as PNN')
  parser.add_argument('--result', type=str, default='SurfProNN/test', help='the same to above')
  parser.add_argument('--runs', type=str, default='SurfProNN', help='the same to above')
  parser.add_argument('--capri', type=bool, default=False, help='test capri')
  parser.add_argument('--use_plm', type=bool, default=True, help='use plm feature')
  return parser.parse_args()

def pc_normalize(pc):
    """
    Normalize coords for all sample points
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class DataLoader(Dataset):
    def __init__(self, root, document, split='train', fold=1, use_res=True, use_more_class=False, use_plm=True):
        self.root = root
        self.document = document
        self.use_res = use_res
        self.use_plm = use_plm
        if self.use_plm:
            self.plm = h5py.File('/home/data/embedding/dockground.h5', 'r')
        self.use_more_class = use_more_class

        self.cat = []
        if self.use_more_class:
            self.catfile = os.path.join(self.root, 'class.txt')  # incorrect,acceptable,medium,high
            self.cat = [line.rstrip() for line in open(self.catfile)]
        else:
            self.cat = ["negtive", "positive"]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        data_paths = {}
        assert (split == 'train' or split == 'valid' or split == 'test' or split == 'finetune')


        if split == 'finetune':
            data_paths[split] = [self.root + '/' + line.split('\t')[0] + ' ' + line.split('\t')[1].rstrip() for line in
                                 open(os.path.join(self.document, f"fold{fold}_train.txt"))]
            data_paths[split].extend([self.root + '/' + line.split('\t')[0] + ' ' + line.split('\t')[1].rstrip() for line in
                                 open(os.path.join(self.document, f"fold{fold}_valid.txt"))])
        else:
            data_paths[split] = [self.root + '/' + line.split('\t')[0] + ' ' + line.split('\t')[1].rstrip() for line in
                             open(os.path.join(self.document, f"fold{fold}_{split}.txt"))]

        self.datapath = data_paths[split]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        data = []
        plm_features = []
        labels = [int(path.split()[1]) for path in self.datapath]
        paths = [path.split()[0] for path in self.datapath]

        if self.use_plm:
            for path in paths:
                key = path.split('/')[-2] + '_' + path.split('/')[-1]
                plm_feature_receptor = self.plm[key + '_A']  # [1024,1]
                plm_feature_ligand = self.plm[key + '_B']
                plm_feature = np.concatenate((plm_feature_receptor, plm_feature_ligand), axis=0)  # [2048,1]
                plm_features.append(plm_feature)

        for index in tqdm(range(len(self.datapath))):  

            path_and_index = self.datapath[index]
            path_and_index = path_and_index.split()
            path = path_and_index[0]
            label = int(path_and_index[1])

            try:
                point_set = np.loadtxt(path + '.txt', delimiter=' ').astype(np.float32)  # [N,3+4+20+2]
                point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
            except:
                continue

            if not self.use_res:
                point_set = np.concatenate((point_set[:, 0:7], point_set[:, -2:]), axis=1)  # only atom

            data.append(point_set)


        self.data = data
        self.labels = labels
        self.paths = paths
        self.plm_feature = plm_features

    def load_txt(self,path,shared_list):
        point_set = np.loadtxt(path + '.txt', delimiter=' ').astype(np.float32)  # [N,3+4+20+2]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_res:
            point_set = np.concatenate((point_set[:, 0:7], point_set[:, -2:]), axis=1)
        shared_list.append(point_set)
    def get_labels(self):
        return self.labels

    def get_size(self):
        return len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        point_set = self.data[index]
        label = self.labels[index]
        plm = self.plm_feature[index]
        return point_set, label, plm
      
def collate_fn(batch):
    f_num = batch[0][0].shape[1]
    max_natoms = 1000
    labels = []
    plms = []
    point_set = np.zeros((len(batch), max_natoms, f_num))

    for i in range(len(batch)):
        item = batch[i]
        num_atom = item[0].shape[0]
        point_set[i, :num_atom] = item[0]
        labels.append(item[1])
        plms.append(item[2])

    labels = np.array(labels)
    plms = np.array(plms)
    return point_set, labels, max_natoms, plms

def train(args, fold=1):
  args = parse_args()

  '''HYPER PARAMETER'''
  # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

  '''CREATE DIR'''
  checkpoint_dir = Path(os.path.join(ROOT_DIR, 'checkpoint'))
  checkpoint_dir.mkdir(exist_ok=True)

  checkpoint_dir = Path(os.path.join(ROOT_DIR, 'checkpoint', args.checkpoint))
  checkpoint_dir.mkdir(exist_ok=True)

  timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
  experiment_dir_root = Path(os.path.join(ROOT_DIR, 'log/'))
  experiment_dir_root.mkdir(exist_ok=True)
  if args.log_dir is None:
      experiment_dir_root = experiment_dir_root.joinpath(timestr)
  else:
      experiment_dir_root = experiment_dir_root.joinpath(f"{args.log_dir}_FOLD{fold}")
  experiment_dir_root.mkdir(exist_ok=True)
  experiment_dir = experiment_dir_root

  logger = logging.getLogger("model")
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  file_handler = logging.FileHandler('%s/%s.txt' % (experiment_dir, args.model))
  file_handler.setLevel(logging.INFO)
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)
  log_string(logger, 'PARAMETER ...')
  log_string(logger, args)

  '''TensorboardX'''
  # defual folder: ./runs
  if not os.path.exists(os.path.join(ROOT_DIR, 'runs', args.runs)):
      os.mkdir(os.path.join(ROOT_DIR, 'runs', args.runs))

  writer_path = os.path.join(ROOT_DIR, 'runs', args.runs, f"FOLD{fold}")
  Path(writer_path).mkdir(exist_ok=True)
  writer = SummaryWriter(writer_path)

  '''DATA LOADING'''
  log_string(logger, 'Load dataset ...')
  DC_PATH = os.path.join(ROOT_DIR, 'document')
  DC_PATH = os.path.join(DC_PATH, args.document_dir)

  DATA_PATH = os.path.join(ROOT_DIR, 'data')
  DATA_PATH = os.path.join(DATA_PATH, args.data_dir)

  TRAIN_DATASET = DataLoader(root=DATA_PATH, document=DC_PATH, fold=fold, split='train', use_res=args.use_res, use_plm=args.use_plm)
  VALID_DATASET = DataLoader(root=DATA_PATH, document=DC_PATH, fold=fold, split='valid', use_res=args.use_res, use_plm=args.use_plm)

  '''MODEL LOADING'''
  log_string(logger, 'Load model ...')
  MODEL = importlib.import_module(args.model)
  shutil.copy(os.path.join(ROOT_DIR, 'model', '%s.py' % args.model), experiment_dir)

  patience = 10
  early_stopping = EarlyStopping(patience, verbose=True)

  '''SEED'''
  provider.set_seed(args.seed)
  logger.info('Set random seed ...')


  classifier = MODEL.SufrProNN().to(device)

  print("The model has {} millions parameterd".format(
      sum(p.numel() for p in classifier.parameters() if p.requires_grad) / 1000000.0))

  criterion = nn.BCELoss()

  try:
      checkpoint = torch.load(
          os.path.join(ROOT_DIR, 'checkpoint', args.checkpoint, "FOLD" + str(fold), "last_model.pth"))
      start_epoch = checkpoint['epoch']
      classifier.load_state_dict(checkpoint['model_state_dict'])
      log_string(logger, 'Use pretrain model')
  except:
      Path(f"{ROOT_DIR}/checkpoint/{args.checkpoint}/FOLD{fold}").mkdir(exist_ok=True)
      log_string(logger, 'No existing model, starting training from scratch...')
      start_epoch = 0

  if args.optimizer == 'Adam':
      optimizer = torch.optim.Adam(
          classifier.parameters(),
          lr=args.learning_rate,
          betas=(0.9, 0.999),
          eps=1e-08,
          weight_decay=args.decay_rate
      )
  else:
      optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

  global_epoch = 0
  global_step = 0
  best_instance_acc = 0.0
  best_class_acc = 0.0

  train_size = TRAIN_DATASET.get_size()
  valid_size = VALID_DATASET.get_size()

  trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, sampler=ImbalancedDatasetSampler(TRAIN_DATASET),
                                                batch_size=args.batch_size, shuffle=False, num_workers=8,
                                                drop_last=False, collate_fn=collate_fn, worker_init_fn=seed_worker)
  validDataLoader = torch.utils.data.DataLoader(VALID_DATASET, sampler=ImbalancedDatasetSampler(VALID_DATASET),
                                                batch_size=args.batch_size, shuffle=False, num_workers=8,
                                                drop_last=False, collate_fn=collate_fn, worker_init_fn=seed_worker)

  # TRANING
  logger.info('Start training...')
  for epoch in range(start_epoch, args.epoch):

      log_string(logger, 'Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

      train_aucs = []
      mean_correct = []
      pred_list = []
      target_list = []
      running_loss = 0.0
      scheduler.step()
      for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
          points, target, npoint, plms = data
          points = provider.random_point_dropout(points)
          points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])

          points = torch.Tensor(points) 
          target = torch.Tensor(target)
          plms = torch.Tensor(plms)

          points = points.transpose(2, 1)
          points, target, plms = points.to(device), target.to(device), plms.to(device)
          optimizer.zero_grad()

          classifier = classifier.train()
          pred = classifier(points)
          loss = criterion(pred, target)

          zero = torch.zeros_like(pred)
          one = torch.ones_like(pred)
          pred_choice = torch.where(pred > 0.5, one, zero)
          correct = pred_choice.eq(target.long().data).cpu().sum()

          mean_correct.append(correct.item() / float(points.size()[0]))
          running_loss += loss.item()

          target_list.extend(target.data.cpu().tolist())
          pred_list.extend(pred_choice.data.cpu().tolist())

          loss.backward()
          optimizer.step()
          global_step += 1

      train_instance_acc = np.mean(mean_correct)
      train_auc = roc_auc_score(target_list, pred_list)
      log_string(logger, 'Train Instance Accuracy: %f' % train_instance_acc)
      log_string(logger, 'Train Instance AUC: %f' % train_auc)

      running_loss = running_loss / train_size
      writer.add_scalar('train_loss', running_loss, global_step=epoch)
      writer.add_scalar('train_auc', train_auc, global_step=epoch)
      writer.add_scalar('train_acc', train_instance_acc, global_step=epoch)

      # Valid
      pred_list = []
      target_list = []
      with torch.no_grad():
          mean_correct = []
          # valid_aucs = []

          running_loss = 0.0
          class_acc = np.zeros((2, 3))
          for j, data in tqdm(enumerate(validDataLoader), total=len(validDataLoader)):
              points, target, npoint,plm = data
              points = torch.Tensor(points)
              target = torch.Tensor(target)
              plm =  torch.Tensor(plm).to(device)
              points = points.transpose(2, 1)
              points, target = points.to(device), target.to(device)
              classifier = classifier.eval()
              pred = classifier(points)
              loss = criterion(pred, target)

              zero = torch.zeros_like(pred)
              one = torch.ones_like(pred)
              pred_choice = torch.where(pred > 0.5, one, zero)

              target_list.extend(target.data.cpu())
              pred_list.extend(pred_choice.data.cpu())

              for cat in np.unique(target.cpu()):
                  classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                  cat = int(cat)
                  class_acc[cat, 0] += classacc.item()
                  class_acc[cat, 1] += points[target == cat].size()[0]

              correct = pred_choice.eq(target.long().data).cpu().sum()
              mean_correct.append(correct.item() / float(points.size()[0]))

              running_loss += loss.item()
          valid_auc = roc_auc_score(target_list, pred_list)

          class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
          class_acc = np.mean(class_acc[:, 2])
          valid_instance_acc = np.mean(mean_correct)
          running_loss = running_loss / valid_size

          writer.add_scalar('valid_loss', running_loss, global_step=epoch)
          writer.add_scalar('valid_acc', valid_instance_acc, global_step=epoch)
          writer.add_scalar('valid_auc', valid_auc, global_step=epoch)

          log_string(logger, 'Valid Instance Accuracy: %f, Class Accuracy: %f, valid AUC: %f' % (
          valid_instance_acc, class_acc, valid_auc))
          log_string(logger, 'Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

          if (class_acc >= best_class_acc and epoch > 5):
              best_instance_acc = valid_instance_acc
              best_class_acc = class_acc
              best_epoch = epoch + 1
              logger.info('Save model...')
              Path(os.path.join('checkpoint', args.checkpoint, "FOLD" + str(fold))).mkdir(exist_ok=True)
              savepath = os.path.join('checkpoint', args.checkpoint, "FOLD" + str(fold), 'best_model.pth')
              log_string(logger, 'Saving at %s' % savepath)
              state = {
                  'epoch': best_epoch,
                  'instance_acc': valid_instance_acc,
                  'class_acc': class_acc,
                  'model_state_dict': classifier.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
              }
              torch.save(state, savepath)
          global_epoch += 1

          savepath = os.path.join(ROOT_DIR, 'checkpoint', args.checkpoint, f"FOLD{fold}", f'epoch_{epoch}.pth')
          log_string(logger, 'Saving at %s' % savepath)
          state = {
              'epoch': epoch,
              'instance_acc': valid_instance_acc,
              'class_acc': class_acc,
              'model_state_dict': classifier.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
          }
          torch.save(state, savepath) 

          if (epoch > 5):
              early_stopping(1 - class_acc, classifier)

          if early_stopping.early_stop:
              print('Early stop')
              break

  logger.info('End of training...')
  logger.removeHandler(file_handler)


def main(args):
  for fold in range(1,5):
      train(args,fold)

if __name__ == '__main__':
  args = parse_args()
  main(args)
