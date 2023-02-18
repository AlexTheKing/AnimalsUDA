import numpy as np
import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List
import itertools
from torch import nn, optim

@dataclass
class TrainMetrics:
    train_loss_history: List[float] = field(default_factory=list)
    train_accuracy_history: List[float] = field(default_factory=list)
    validation_accuracy_history: List[float] = field(default_factory=list)
    best_validation_accuracy: float = 0
    best_validation_loss: float = 1e6

@dataclass
class EpochMetrics:
    supervised_loss: float = 0
    unsupervised_loss: float = 0
    epoch_loss: float = 0
    correct_samples: int = 0
    total_samples: int = 0
    batches_count: int = 0


class UDATrainer:
    def __init__(self, model, uda_data,
                 optimizer=None, scheduler=None,
                 device=torch.device('cuda'), tsa_schedule='exp_schedule'):
        self.model = model
        self.uda_data = uda_data
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            reduction='none'
            #             label_smoothing=0.2  # TODO: label smooting??
        )
        self.kldiv_loss = nn.KLDivLoss(reduction='none')
        self.optimizer = (
            optimizer
            if optimizer
            else optim.SGD(
                model.parameters(), lr=0.0001,
                momentum=0.9, weight_decay=5e-4
            )
        )
        self.scheduler = (
            scheduler
            if scheduler
            else optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                100,
                verbose=True
            )
        )
        self.metrics = TrainMetrics()
        self.device = device
        self.tsa_schedule = tsa_schedule
        steps_per_epoch = np.ceil(len(self.uda_data.train_indices) / self.uda_data.supervised_batch_size)
        self.tsa_steps_to_be_free = steps_per_epoch * 100  # TODO: adjust!!!
        self.uda_softmax_temp = 0.4  # from paper 0.4
        self.uda_confidence_threshold = 0.4  # from paper 0.5
        self.unsupervised_coefficient = 1  # from paper 1
        print('TSA Steps to be free', self.tsa_steps_to_be_free)
        print('UDA Softmax Temp', self.uda_softmax_temp)
        print('UDA Confidence Threshold', self.uda_confidence_threshold)
        print('UDA Unsupervised coefficient', self.unsupervised_coefficient)

    def train(self, early_stopping_epochs=None, validation_epoch=2, use_amp=True):
        print(f'Use AMP is {use_amp}')
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        early_stopping_active_epochs = 0
        global_step = 0
        for epoch in range(10**6):
            self.model.train()
            epoch_metrics = EpochMetrics()
            train_supervised_loader, train_unsupervised_loader, validation_loader = self.uda_data.sample_loaders()
            train_unsupervised_loader_iterator = itertools.cycle(train_unsupervised_loader)
            for batch_index, (supervised_images, supervised_targets) in enumerate(tqdm(train_supervised_loader)):
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    unsupervised_original_images, unsupervised_augmented_images = next(
                        train_unsupervised_loader_iterator
                    )
                    all_images = torch.concat([
                        supervised_images,
                        unsupervised_original_images,
                        unsupervised_augmented_images
                    ])
                    all_images_gpu = all_images.to(self.device)
                    supervised_targets_gpu = supervised_targets.to(self.device)
                    all_logits = self.model(all_images_gpu)
                    supervised_batch_size = supervised_images.shape[0]
                    supervised_logits = all_logits[:supervised_batch_size]
                    supervised_loss = self.cross_entropy_loss(supervised_logits, supervised_targets_gpu)

                    # TSA & Supervised Loss
                    supervised_loss, avg_supervised_loss = self.anneal_supervised_loss(
                        supervised_logits, supervised_targets_gpu, supervised_loss, global_step
                    )
                    total_loss = avg_supervised_loss
                    epoch_metrics.supervised_loss += float(avg_supervised_loss)

                    # Unsupervised Loss
                    augment_batch_size = unsupervised_original_images.shape[0]
                    original_images_logits = all_logits[supervised_batch_size: supervised_batch_size + augment_batch_size]
                    augmented_images_logits = all_logits[supervised_batch_size + augment_batch_size:]
                    original_images_logits_temperatured = (original_images_logits / self.uda_softmax_temp).detach()

                    # KL Loss
                    original_images_temperatured_probs = nn.functional.softmax(original_images_logits_temperatured, dim=-1)
                    augmented_images_probs = nn.functional.log_softmax(augmented_images_logits, dim=-1)
                    augmentation_loss = self.kldiv_loss(augmented_images_probs, original_images_temperatured_probs).sum(-1)

                    # UDA Confidence Threshold
                    largest_prob, _ = original_images_temperatured_probs.max(-1)
                    loss_mask = (largest_prob > self.uda_confidence_threshold).int().detach()
                    augmentation_loss *= loss_mask

                    # Finally
                    avg_unsupervised_loss = augmentation_loss.mean()
                    epoch_metrics.unsupervised_loss += float(avg_unsupervised_loss)
                    total_loss += self.unsupervised_coefficient * avg_unsupervised_loss

                self.optimizer.zero_grad(set_to_none=True)
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                supervised_predictions = nn.functional.softmax(supervised_logits, dim=-1)
                _, predictions_indices = torch.max(supervised_predictions, 1)
                _, target_indices = torch.max(supervised_targets_gpu, 1)

                epoch_metrics.correct_samples += torch.sum(predictions_indices == target_indices)
                epoch_metrics.total_samples += supervised_targets.shape[0]
                epoch_metrics.epoch_loss += float(total_loss)
                epoch_metrics.batches_count += 1
                global_step += 1

            self.scheduler.step()

            train_loss = epoch_metrics.epoch_loss / epoch_metrics.batches_count
            train_accuracy = float(epoch_metrics.correct_samples) / epoch_metrics.total_samples

            validation_accuracy_to_be_checked = epoch % validation_epoch == 0
            if validation_accuracy_to_be_checked:
                print('Checking validation accuracy...')
                validation_accuracy, validation_loss = self.compute_accuracy(validation_loader)
            else:
                validation_accuracy, validation_loss = (
                    self.metrics.best_validation_accuracy, self.metrics.best_validation_loss
                )

            self.metrics.train_loss_history.append(train_loss)
            self.metrics.train_accuracy_history.append(train_accuracy)
            self.metrics.validation_accuracy_history.append(validation_accuracy)

            if validation_loss < self.metrics.best_validation_loss:
                torch.save(self.model.state_dict(), 'best_model.pth')
                self.metrics.best_validation_accuracy = validation_accuracy
                self.metrics.best_validation_loss = validation_loss
                print(f'Saved model for validation accuracy: {self.metrics.best_validation_accuracy:.5f}, '
                      f'loss {self.metrics.best_validation_loss:.5f}')
                early_stopping_active_epochs = 0
            else:
                early_stopping_active_epochs += 1

            print(f'EPOCH {epoch + 1}\n'
                  f'Train accuracy: {train_accuracy:.5f}\n'
                  f'Validation accuracy: {validation_accuracy:.5f} '
                  f'{"(actual)" if validation_accuracy_to_be_checked else "(best achieved)"}\n'
                  f'Train avg loss: {train_loss:.5f}\n'
                  f'Validation avg loss: {validation_loss:.5f} '
                  f'{"(actual)" if validation_accuracy_to_be_checked else "(best achieved)"}\n',
                  f'Supervised avg loss: {(epoch_metrics.supervised_loss / epoch_metrics.batches_count):.5f}\n'
                  f'Unsupervised avg loss: {(epoch_metrics.unsupervised_loss / epoch_metrics.batches_count):.5f}\n')

            if early_stopping_active_epochs >= early_stopping_epochs:
                print(f'Early stopping activated! '
                      f'Best accuracy: {self.metrics.best_validation_accuracy:.5f}, '
                      f'best loss {self.metrics.best_validation_loss:.5f}')
                break

    def get_tsa_threshold(self, global_step, start, end):
        step_ratio = float(global_step) / self.tsa_steps_to_be_free
        if self.tsa_schedule == "linear_schedule":
            coefficient = step_ratio
        elif self.tsa_schedule == "exp_schedule":
            scale = 5
            # [exp(-5), exp(0)] = [1e-2, 1]
            coefficient = np.exp((step_ratio - 1) * scale)
        elif self.tsa_schedule == "log_schedule":
            scale = 5
            # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
            coefficient = 1 - np.exp((-step_ratio) * scale)
        else:
            raise ValueError('Unknown TSA schedule')
        return coefficient * (end - start) + start

    def anneal_supervised_loss(self, supervised_logits, supervised_targets, supervised_loss, global_step):
        tsa_start = 1. / supervised_targets.shape[1]
        effective_train_prob_threshold = self.get_tsa_threshold(global_step, start=tsa_start, end=1)

        supervised_probs = nn.functional.softmax(supervised_logits, dim=-1)
        correct_label_probs = (supervised_targets * supervised_probs).sum(-1)
        larger_than_threshold = correct_label_probs > effective_train_prob_threshold
        loss_mask = (1 - larger_than_threshold.int()).detach()
        supervised_loss *= loss_mask
        avg_supervised_loss = (supervised_loss.sum() / torch.maximum(loss_mask.sum(), torch.tensor([1], device=self.device)))
        return supervised_loss, avg_supervised_loss

    def compute_accuracy(self, loader):
        self.model.eval()
        epoch_metrics = EpochMetrics()
        cross_entropy_loss = nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, targets in tqdm(loader):
                epoch_metrics.batches_count += 1
                images_gpu = images.to(self.device)
                targets_gpu = targets.to(self.device)
                logits = self.model(images_gpu)

                epoch_metrics.epoch_loss += float(cross_entropy_loss(logits, targets_gpu))

                _, predictions_indices = torch.max(nn.functional.softmax(logits), 1)
                _, target_indices = torch.max(targets_gpu, 1)
                epoch_metrics.correct_samples += torch.sum(predictions_indices == target_indices)
                epoch_metrics.total_samples += targets.shape[0]

        return (
            float(epoch_metrics.correct_samples) / epoch_metrics.total_samples,
            epoch_metrics.epoch_loss / epoch_metrics.batches_count
        )
