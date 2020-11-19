#!/usr/bin/env python
# coding:utf-8
"""
Tencent is pleased to support the open source community by making NeuralClassifier available.
Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the MIT License (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for thespecific language governing permissions and limitations under
the License.
"""
import itertools

import torch
import torch.nn as nn
import numpy as np
from dataset_preprocessing.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from model.cnn import CNNLayer
from model.embedding import PositionEmbedding


class PositionalCNN(Classifier):
    def __init__(self, dataset, config):
        super(PositionalCNN, self).__init__(dataset, config)

        self.pad = dataset.token_map[dataset.VOCAB_PADDING]

        seq_max_len = config.feature.max_token_len

        self.position_enc = PositionEmbedding(seq_max_len,
                                              config.embedding.dimension,
                                              self.pad)

        self.layer_stack = nn.ModuleList([CNNLayer(config.embedding.dimension,  config.PositionalCNN.kernel_size)])

        hidden_size = config.embedding.dimension
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size//2)
        self.linear2 = torch.nn.Linear(hidden_size//2, len(dataset.label_map))

        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def get_parameter_optimizer_dict(self):
        params = list()

        # for tn in self.token_embedding.embedding.weight.data:
        #     print(len(tn))
        #
        # labels = []
        # for key, val in self.token_embedding.embedding.data_dic.items():
        #     labels.append(key)
        #
        # count = 0
        # # max_count=itertools.product('ACGT', repeat=self.config.feature.max_char_len_per_token )
        # # X = np.zeros(shape=(self.token_embedding.embedding.weight.data.shape[0], self.token_embedding.embedding.weight.data.shape[1]))
        # max_count=100
        #
        # X = np.zeros(shape=(max_count, self.token_embedding.embedding.weight.data.shape[1]))
        # for tn in self.token_embedding.embedding.weight.data:
        #     print(len(tn))
        #     X[count] = tn
        #     count += 1
        #     if count >= max_count: break
        #
        # # It is recommended to use PCA first to reduce to ~50 dimensions
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=50)
        # X_50 = pca.fit_transform(X)
        #
        # # Using TSNE to further reduce to 2 dimensions
        # from sklearn.manifold import TSNE
        # model_tsne = TSNE(n_components=2, random_state=0)
        # Y = model_tsne.fit_transform(X_50)
        #
        # # Show the scatter plot
        # import matplotlib.pyplot as plt
        # print(Y[:, 0])
        #
        # plt.scatter(Y[:, 0], Y[:, 1], 10)
        #
        # # Add labels
        # for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
        #     plt.annotate(label, xy=(x, y), xytext=(-15, 0), textcoords='offset points', size=12)
        #
        # # plt.xlim([-200, -400])
        # plt.show()

        #

        params.append({'params': self.token_embedding.parameters()})
        for i in range(0, len(self.layer_stack)):
            params.append({'params': self.layer_stack[i].parameters()})
        params.append({'params': self.linear1.parameters()})
        params.append({'params': self.linear2.parameters()})

        return params

    def update_lr(self, optimizer, epoch):
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0

    def forward(self, batch):


        src_seq = batch[cDataset.DOC_TOKEN].to(self.config.device)
        embedding = self.token_embedding(src_seq)


        # Prepare masks
        batch_lens = (src_seq != self.pad).sum(dim=-1)
        src_pos = torch.zeros_like(src_seq, dtype=torch.long)
        for row, length in enumerate(batch_lens):
            src_pos[row][:length] = torch.arange(1, length + 1)
        emb_output =embedding + self.position_enc(src_pos)
        # enc_output =self.position_enc(src_pos)


        for layer in self.layer_stack:
                emb_output = layer(emb_output)
        emb_output = torch.mean(emb_output, 1)

        return self.linear2(self.linear1(emb_output))
