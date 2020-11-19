from eval import *
import os
import shutil
import sys
import time
import csv
import torch
from torch.utils.data import DataLoader
from colorama import Fore
import util
from config import Config
from dataset_preprocessing.classification_dataset import ClassificationDataset
from dataset_preprocessing.collator import ClassificationCollator
from dataset_preprocessing.collator import FastTextCollator
from dataset_preprocessing.collator import ClassificationType
from evaluate.classification_evaluate import \
    ClassificationEvaluator as cEvaluator

from model.classification.textrnn import TextRNN
from model.classification.positionalcnn import PositionalCNN
from model.loss import ClassificationLoss
from model.model_util import get_optimizer
from util import ModeType
# from torchviz import make_dot



def get_data_loader(dataset_name, collate_name, conf):
    """Get data loader: Train, Validate, Test
    """
    train_dataset = globals()[dataset_name](
        conf, conf.data.train_json_files, generate_dict=True)
    collate_fn = globals()[collate_name](conf, len(train_dataset.label_map))

    train_data_loader = DataLoader(
        train_dataset, batch_size=conf.train.batch_size, shuffle=True,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    validate_dataset = globals()[dataset_name](
        conf, conf.data.validate_json_files)
    validate_data_loader = DataLoader(
        validate_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    test_dataset = globals()[dataset_name](conf, conf.data.test_json_files)
    test_data_loader = DataLoader(
        test_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    return train_data_loader, validate_data_loader, test_data_loader


def get_classification_model(model_name, dataset, conf):
    """Get classification model from configuration
    """
    model = globals()[model_name](dataset, conf)
    model = model.cuda(conf.device) if conf.device.startswith("cuda") else model
    return model


class ClassificationTrainer(object):
    def __init__(self, label_map, logger, evaluator, conf, loss_fn):
        self.label_map = label_map
        self.logger = logger
        self.evaluator = evaluator
        self.conf = conf
        self.loss_fn = loss_fn
        # if self.conf.task_info.hierarchical:
        #     self.hierar_relations = get_hierar_relations(
        #         self.conf.task_info.hierar_taxonomy, label_map)

    def train(self, data_loader, model, optimizer, stage, epoch):
        model.update_lr(optimizer, epoch)
        model.train()
        return self.run(data_loader, model, optimizer, stage, epoch,
                        ModeType.TRAIN)

    def eval(self, data_loader, model, optimizer, stage, epoch):
        model.eval()
        return self.run(data_loader, model, optimizer, stage, epoch)

    def run(self, data_loader, model, optimizer, stage,
            epoch, mode=ModeType.EVAL):
        is_multi = True
        predict_probs = []
        standard_labels = []
        num_batch = data_loader.__len__()
        total_loss = 0.
        for batch in data_loader:
            logits = model(batch)
            # print(make_dot(logits, params=dict(model.named_parameters())))
            # hierarchical classification
            loss = self.loss_fn(
                    logits,
                    batch[ClassificationDataset.DOC_LABEL].to(self.conf.device),
                    False,
                    is_multi)
            if mode == ModeType.TRAIN:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                continue
            total_loss += loss.item()
            if not is_multi:
                result = torch.nn.functional.softmax(logits, dim=1).cpu().tolist()
            else:
                result = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(result)
            standard_labels.extend(batch[ClassificationDataset.DOC_LABEL_LIST])
        if mode == ModeType.EVAL:
            total_loss = total_loss / num_batch
            (_, precision_list, recall_list, fscore_list, right_list,
             predict_list, standard_list) = \
                self.evaluator.evaluate(
                    predict_probs, standard_label_ids=standard_labels, label_map=self.label_map,
                    threshold=self.conf.eval.threshold, top_k=self.conf.eval.top_k,
                     is_multi=is_multi)

            self.logger.warn(
                "%s performance at epoch %d is precision: %f, "
                "recall: %f, fscore: %f, macro-fscore: %f, right: %d, predict: %d, standard: %d.\n"
                "Loss is: %f." % (
                    stage, epoch, precision_list[0][cEvaluator.MICRO_AVERAGE],
                    recall_list[0][cEvaluator.MICRO_AVERAGE],
                    fscore_list[0][cEvaluator.MICRO_AVERAGE],
                    fscore_list[0][cEvaluator.MACRO_AVERAGE],
                    right_list[0][cEvaluator.MICRO_AVERAGE],
                    predict_list[0][cEvaluator.MICRO_AVERAGE],
                    standard_list[0][cEvaluator.MICRO_AVERAGE], total_loss))
            return precision_list[0][cEvaluator.MICRO_AVERAGE], precision_list[0][cEvaluator.MACRO_AVERAGE],recall_list[0][cEvaluator.MICRO_AVERAGE],recall_list[0][cEvaluator.MACRO_AVERAGE], fscore_list[0][cEvaluator.MICRO_AVERAGE],fscore_list[0][cEvaluator.MACRO_AVERAGE]




def load_checkpoint(file_name, conf, model, optimizer):
    checkpoint = torch.load(file_name)
    conf.train.start_epoch = checkpoint["epoch"]
    best_performance = checkpoint["best_performance"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return best_performance


def save_checkpoint(state, file_prefix):
    file_name = file_prefix + "_" + str(state["epoch"])
    torch.save(state, file_name)


def train(conf, dir, fold,ngram):
    sum_precision = []
    sum_recall = []
    sum_accuracy = []
    sum_f1_scor = []
    sum_micro_fscore = []
    sum_macro_fscore = []
    sum_hamming_loss = []
    sum_averagePrecision = []
    for i in range(fold):

        print("______________________Fold",i,"______________________")
        conf.data.train_json_files=[os.path.join(dir, str(i),"train.json")]
        conf.data.test_json_files=[os.path.join(dir, str(i),"test.json")]
        conf.data.validate_json_files=[os.path.join(dir, str(i),"valid.json")]

        logger = util.Logger(conf)
        if not os.path.exists(conf.checkpoint_dir):
            os.makedirs(conf.checkpoint_dir)

        model_name = conf.model_name
        dataset_name = "ClassificationDataset"
        collate_name = "FastTextCollator" if model_name == "FastText" \
            else "ClassificationCollator"
        train_data_loader, validate_data_loader, test_data_loader = \
            get_data_loader(dataset_name, collate_name, conf)
        empty_dataset = globals()[dataset_name](conf, [])
        model = get_classification_model(model_name, empty_dataset, conf)
        loss_fn = globals()["ClassificationLoss"](
            label_size=len(empty_dataset.label_map), loss_type=conf.train.loss_type)
        optimizer = get_optimizer(conf, model)
        evaluator = cEvaluator(conf.eval.dir)
        trainer = globals()["ClassificationTrainer"](
            empty_dataset.label_map, logger, evaluator, conf, loss_fn)

        best_epoch = -1
        best_performance = 0
        model_file_prefix = conf.checkpoint_dir + "/" + model_name
        for epoch in range(conf.train.start_epoch,
                           conf.train.start_epoch + conf.train.num_epochs):
            start_time = time.time()
            trainer.train(train_data_loader, model, optimizer, "Train", epoch)
            trainer.eval(train_data_loader, model, optimizer, "Train", epoch)
            performance = trainer.eval(
                validate_data_loader, model, optimizer, "Validate", epoch)
            trainer.eval(test_data_loader, model, optimizer, "test", epoch)
            if performance[4] > best_performance:  # record the best model
                best_epoch = epoch
                best_performance = performance[4]
            save_checkpoint({
                'epoch': epoch,
                'model_name': model_name,
                'state_dict': model.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimizer.state_dict(),
            }, model_file_prefix)
            time_used = time.time() - start_time
            logger.info("Epoch %d cost time: %d second" % (epoch, time_used))

        # best model on validateion set
        best_epoch_file_name = model_file_prefix + "_" + str(best_epoch)
        best_file_name = model_file_prefix + "_best"
        shutil.copyfile(best_epoch_file_name, best_file_name)

        load_checkpoint(model_file_prefix + "_" + str(best_epoch), conf, model,
                        optimizer)
        trainer.eval(test_data_loader, model, optimizer, "Best test", best_epoch)

        evaluation_measures=kfold_eval(config)
        sum_precision.append(evaluation_measures["Precision"])
        sum_recall.append(evaluation_measures["Recall"])
        sum_accuracy.append(evaluation_measures["Accuracy"])
        sum_f1_scor.append(evaluation_measures["F1 score"])
        sum_micro_fscore.append(evaluation_measures["f-1 Micro"])
        sum_macro_fscore.append(evaluation_measures["f-1 Macro"])
        sum_hamming_loss.append(evaluation_measures["Hamming Loss"])
        sum_averagePrecision.append(evaluation_measures["averagePrecision"])


        shutil.rmtree(conf.eval.model_dir.split("/")[0])


    print("_________________________________kfolds Metrics____________________________________")
    print(Fore.BLUE + "k-fold  precision", sum(sum_precision) / fold)
    print(Fore.RED + str(sum_precision))
    print(Fore.BLUE + "k-fold recall", sum(sum_recall) / fold)
    print(Fore.RED + str(sum_recall))
    print(Fore.BLUE + "k-fold fscore", sum(sum_f1_scor) / fold)
    print(Fore.RED + str(sum_f1_scor))
    print(Fore.BLUE + "k-fold Micro Fscore", sum(sum_micro_fscore) / fold)
    print(Fore.RED + str(sum_micro_fscore))
    print(Fore.BLUE + "k-fold Macro Fscore", sum(sum_macro_fscore) / fold)
    print(Fore.RED + str(sum_macro_fscore))
    print(Fore.BLUE + "k-fold Accuracy", sum(sum_accuracy) / fold)
    print(Fore.RED + str(sum_accuracy))
    print(Fore.BLUE + "k-fold Hamming Loss", sum(sum_hamming_loss) / fold)
    print(Fore.RED + str(sum_hamming_loss))
    print(Fore.BLUE + "k-fold averagePrecision", sum(sum_averagePrecision) / fold)
    print(Fore.RED + str(sum_averagePrecision))

    with open('cnn_results', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([str(ngram) + "Grams"])
        writer.writerow(
            ["rand", str(round(sum(sum_precision) / fold, 4)), str(round(sum(sum_recall) / fold,4)), str(round(sum(sum_f1_scor) / fold,4)),
             str(round(sum(sum_micro_fscore) / fold,4)), str(round(sum(sum_macro_fscore) / fold,4)), str(round(sum(sum_accuracy) / fold,4)),
             str(round(sum(sum_hamming_loss) / fold,4)), str(round(sum(sum_averagePrecision) / fold,4))])

if __name__ == '__main__':
    kfold = 10
    ngrams=4
    # for ngrams in range(3,5):
    config = Config(config_file=sys.argv[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.train.visible_device_list)
    torch.manual_seed(2019)
    torch.cuda.manual_seed(2019)
    config.feature.max_char_len_per_token = ngrams
    train(config, "MiRNA_dataset/" + str(ngrams), kfold, ngrams)

