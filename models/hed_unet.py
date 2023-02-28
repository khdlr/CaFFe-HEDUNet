import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from einops import rearrange

from .layers import Convx2, DownBlock, UpBlock


class HEDUNet(pl.LightningModule):
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)

    self.zones_metric = metric = torchmetrics.JaccardIndex(task='multiclass', num_classes=4, average=None)
    self.fronts_metric = torchmetrics.JaccardIndex(task='multiclass', num_classes=2, average=None)

    stack_height = 5
    self.output_channels = 5

    self.deep_supervision = True

    bc = 16
    self.init = nn.Conv2d(1, bc, 1)

    conv_args = dict(
      conv_block=Convx2,
      bn=True,
      padding_mode='replicate'
    )

    self.down_blocks = nn.ModuleList([
      DownBlock((1<<i)*bc, (2<<i)*bc, **conv_args)
      for i in range(stack_height)
    ])

    self.up_blocks = nn.ModuleList([
      UpBlock((2<<i)*bc, (1<<i)*bc, **conv_args)
      for i in reversed(range(stack_height))
    ])

    self.predictors = nn.ModuleList([
      nn.Conv2d((1<<i)*bc, self.output_channels, 1)
      for i in reversed(range(stack_height + 1))
    ])

    self.queries = nn.ModuleList([
      nn.Conv2d((1<<i)*bc, 1, 1)
      for i in reversed(range(stack_height + 1))
    ])

  def forward(self, x):
    B, _, H, W = x.shape
    x = self.init(x)

    skip_connections = []
    for block in self.down_blocks:
      skip_connections.append(x)
      x = block(x)

    multilevel_features = [x]
    for block, skip in zip(self.up_blocks, reversed(skip_connections)):
      x = block(x, skip)
      multilevel_features.append(x)

    predictions_list = []
    full_scale_preds = []
    for feature_map, predictor in zip(multilevel_features, self.predictors):
      prediction = predictor(feature_map)
      predictions_list.append(prediction)
      full_scale_preds.append(F.interpolate(prediction, size=(H, W), mode='bilinear', align_corners=True))

    predictions = torch.cat(full_scale_preds, dim=1)

    queries = [F.interpolate(q(feat), size=(H, W), mode='bilinear', align_corners=True)
        for q, feat in zip(self.queries, multilevel_features)]
    queries = torch.cat(queries, dim=1)
    queries = queries.reshape(B, -1, 1, H, W)
    attn = F.softmax(queries, dim=1)
    predictions = predictions.reshape(B, -1, self.output_channels, H, W)
    combined_prediction = torch.sum(attn * predictions, dim=1)

    if self.deep_supervision:
      return combined_prediction, list(reversed(predictions_list))
    else:
      return combined_prediction

  def adapt_mask(self, y):
    mask_type = torch.float32 if self.n_classes == 1 else torch.long
    y = y.squeeze(1)
    y = y.type(mask_type)
    return y

  def give_prediction_for_batch(self, batch):
    x, y, x_name, y_names = batch
    # Safety check
    # if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)) or \
    #     torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
    #   print(f"invalid input detected: x {x}, y {y}", file=sys.stderr)

    y_hat = self.forward(x)

    # Safety check
    # if torch.any(torch.isnan(y_hat)) or torch.any(torch.isinf(y_hat)):
    # print(f"invalid output detected: y_hat {y_hat}", file=sys.stderr)

    return y_hat

  def calc_loss(self, preds, y):
    y_hat_log, deep_supervision_preds = preds
    
    masks, edges = get_pyramid(y, len(deep_supervision_preds))
    main_loss = loss_level(y_hat_log, masks[0], edges[0])
    deep_supervision_losses = list(map(loss_level, deep_supervision_preds, masks, edges))
    full_loss = torch.sum(torch.stack(sum(deep_supervision_losses, start=main_loss)))

    loss_terms = {
      'loss': full_loss, 
      'Seg Loss': main_loss[0],
      'Edge Loss': main_loss[1],
    }
    for i, lvl in enumerate(deep_supervision_losses):
      loss_terms[f'DS Seg Loss@{i}'] = lvl[0]
      loss_terms[f'DS Edge Loss@{i}'] = lvl[1]
    
    seg_metrics = self.zones_metric(y_hat_log[:, :-1].argmax(dim=1), y[:, 0])
    edge_metrics = self.fronts_metric(y_hat_log[:, -1] >= 0, y[:, 1])

    loss_terms['IoU'] = torch.mean(seg_metrics)
    loss_terms['IoU NA Area'] = seg_metrics[0]
    loss_terms['IoU Stone'] = seg_metrics[1]
    loss_terms['IoU Glacier'] = seg_metrics[2]
    loss_terms['IoU Ocean and Ice Melange'] = seg_metrics[3]

    loss_terms['IoU Fronts (mIoU)'] = torch.mean(edge_metrics)
    loss_terms['IoU Background'] = edge_metrics[0]
    loss_terms['IoU Front'] = edge_metrics[1]

    return full_loss, loss_terms

  def make_batch_dictionary(self, loss, metric, name_of_loss):
    """ Give batch_dictionary corresponding to the number of metrics for zone segmentation """
    return metric

  def log_metric(self, outputs, mode):
    # calculating average metric

    avgs = {metric: torch.stack([x[metric] for x in outputs]).mean() for metric in outputs[0]}

    for metric, avg in avgs.items():
      if metric.startswith('IoU'):
        self.logger.experiment.add_scalar(f'IoUs/{mode}/{metric.replace(" ", "_")}', avg, self.current_epoch)
      else:
        self.logger.experiment.add_scalar(f'Losses/{mode}/{metric}', avg, self.current_epoch)

    if mode == "Val":
        self.log('avg_metric_validation', avgs['IoU'])

  def training_step(self, batch, batch_idx):
    x, y, x_name, y_names = batch
    y_hat = self.give_prediction_for_batch(batch)
    train_loss, metric = self.calc_loss(y_hat, y)

    self.log('train_loss', train_loss)
    return self.make_batch_dictionary(train_loss, metric, "loss")

  def training_epoch_end(self, outputs):
    # calculating average loss
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    # logging using tensorboard logger
    self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
    self.log_metric(outputs, "Train")

  def validation_step(self, batch, batch_idx):
    x, y, x_name, y_names = batch
    y_hat = self.give_prediction_for_batch(batch)
    val_loss, metric = self.calc_loss(y_hat, y)
    self.log('val_loss', val_loss)
    return self.make_batch_dictionary(val_loss, metric, "val_loss")

  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    self.logger.experiment.add_scalar("Loss/Val", avg_loss, self.current_epoch)
    self.log_metric(outputs, "Val")
    self.log('avg_loss_validation', avg_loss)

  def test_step(self, batch, batch_idx):
    x, y, x_name, y_names = batch
    y_hat = self.give_prediction_for_batch(batch)
    test_loss, metric = self.calc_loss(y_hat, y)
    self.log('test_loss', test_loss)
    return self.make_batch_dictionary(test_loss, metric, "test_loss")

  def test_epoch_end(self, outputs):
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    # logging using tensorboard logger
    self.logger.experiment.add_scalar("Loss/Test", avg_loss, self.current_epoch)
    self.log_metric(outputs, "Test")

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters())
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                            base_lr=self.hparams.base_lr,
                            max_lr=self.hparams.max_lr,
                            cycle_momentum=False,
                            step_size_up=30000)

    # Workaround from https://github.com/pytorch/pytorch/issues/88684#issuecomment-1307758674
    # is needed to avoid a TypeError when pickling the state_dict
    scheduler._scale_fn_custom = scheduler._scale_fn_ref()
    scheduler._scale_fn_ref = None

    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step'
    }
    return [optimizer], [scheduler_dict]

  @staticmethod
  def add_model_specific_args(parent_parser):
    # Copied from base implementation
    parser = parent_parser.add_argument_group("HEDUNet")
    parser.add_argument('--base_lr', default=4e-5)
    parser.add_argument('--max_lr', default=2e-4)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--kernel_size', type=int, default=3)

    # Hyperparameters for augmentation
    parser.add_argument('--bright', default=0.1, type=float)
    parser.add_argument('--wrap', default=0.1, type=float)
    parser.add_argument('--noise', default=0.5, type=float)
    parser.add_argument('--rotate', default=0.5, type=float)
    parser.add_argument('--flip', default=0.3, type=float)

    return parent_parser


def loss_level(y_hat_log, mask, edges):
  seg_pred, edge_pred = torch.split(y_hat_log, [4, 1], dim=1)

  seg_loss = -torch.mean(torch.sum(F.log_softmax(seg_pred, dim=1) * mask, dim=1))
  edge_loss = auto_weight_bce(edge_pred, edges)

  return [seg_loss, edge_loss]


def auto_weight_bce(y_hat_log, y):
  with torch.no_grad():
    beta = y.mean(dim=[2, 3], keepdims=True)
  logit_1 = F.logsigmoid(y_hat_log)
  logit_0 = F.logsigmoid(-y_hat_log)
  return torch.mean(-(1 - beta) * logit_1 * y \
     - beta * logit_0 * (1 - y))


@torch.no_grad()
def get_pyramid(mask, stack_height): 
  zones, fronts = mask.split([1, 1], dim=1)
  zones = F.one_hot(zones.long(), num_classes=4)
  zones = rearrange(zones, 'B 1 H W C -> B C H W')
  fronts = fronts.float()

  zones_masks = [zones]
  front_masks = [fronts]
  ## Build mip-maps
  for _ in range(stack_height):
    # Pretend we have a batch
    big_mask = zones_masks[-1].float()
    small_mask = F.avg_pool2d(big_mask, 2)
    zones_masks.append(small_mask)

  for _ in range(stack_height):
    big_mask = front_masks[-1]
    small_mask = F.max_pool2d(big_mask, 2)
    front_masks.append(small_mask)

  return zones_masks, front_masks
