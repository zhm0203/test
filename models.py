import os, pdb
import sys

import math
import collections
import json
from typing import Dict, Optional, Any, Union, Callable, List

from loguru import logger
from transformers import BertTokenizer, BertTokenizerFast
import torch
from torch import nn
from torch import Tensor
import torch.nn.init as nn_init
import torch.nn.functional as F
import numpy as np
import pandas as pd
# from transformers import BertTokenizer, BertModel
import constants


pd.set_option('display.max_columns', None)


class TabIDSWordEmbedding(nn.Module):
    r"""
    Encode tokens drawn from column names, categorical and binary features.
    """

    def __init__(self,
                 vocab_size,
                 hidden_dim,
                 padding_idx=0,
                 hidden_dropout_prob=0,
                 layer_norm_eps=1e-5,
                 ) -> None:
        super().__init__()
        # self.tokenizer = BertTokenizer.from_pretrained('./tokenizer/bert-base-uncased')
        # self.bert_model = BertModel.from_pretrained('./tokenizer/bert-base-uncased')
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        nn_init.kaiming_normal_(self.word_embeddings.weight)
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids) -> Tensor:
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TabIDSNumEmbedding(nn.Module):     # hidden_dim=128
    r"""
    Encode tokens drawn from column names and the corresponding numerical features.
    """

    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_bias = nn.Parameter(Tensor(1, 1, hidden_dim))  # add bias    [1,1,128]
        nn_init.uniform_(self.num_bias, a=-1 / math.sqrt(hidden_dim), b=1 / math.sqrt(hidden_dim))

    def forward(self, num_col_emb, x_num_ts, num_mask=None) -> Tensor:
        """
        args:
        num_col_emb: numerical column embedding, (# numerical columns, emb_dim)  每一个列名的嵌入向量的向量
        x_num_ts: numerical features, (bs, emb_dim)
        num_mask: the mask for NaN numerical features, (bs, # numerical columns)
        """
        num_col_emb = num_col_emb.unsqueeze(0).expand((x_num_ts.shape[0], -1, -1))   # 列名嵌入
        # print(num_col_emb.shape)   # (batch_size, num_columns, emb_dim)  torch.Size([64, 78, 128])
        # print(x_num_ts.unsqueeze(-1).shape)      # (batch_size, emb_dim, 1) torch.Size([64, 82, 1])
        # print(self.num_bias.shape)
        num_feat_emb = num_col_emb * x_num_ts.unsqueeze(-1).float() + self.num_bias   # 联合嵌入
        return num_feat_emb


class TabIDSFeatureExtractor:
    r"""
    Process input dataframe to input indices towards transtab encoder,
    usually used to build dataloader for paralleling loading.
    """

    def __init__(self,
                 categorical_columns=None,
                 numerical_columns=None,
                 binary_columns=None,
                 disable_tokenizer_parallel=False,
                 ignore_duplicate_cols=False,
                 **kwargs,
                 ) -> None:
        """
        args:
        categorical_columns: a list of categories feature names
        numerical_columns: a list of numerical feature names
        binary_columns: a list of yes or no feature names, accept binary indicators like
            (yes,no); (true,false); (0,1).
        disable_tokenizer_parallel: true if uses extractor for collator function in torch.DataLoader
        ignore_duplicate_cols: check if exists one col belongs to both cat/num or cat/bin or num/bin,
            if set `true`, the duplicate cols will be deleted, else throws errors.
        """
        if os.path.exists('./tokenizer'):         
            self.tokenizer = BertTokenizerFast.from_pretrained('./tokenizer')
        
        self.tokenizer.__dict__['model_max_length'] = 512
        if disable_tokenizer_parallel:  # disable tokenizer parallel
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns
        self.ignore_duplicate_cols = ignore_duplicate_cols

        if categorical_columns is not None:
            self.categorical_columns = list(set(categorical_columns))
        if numerical_columns is not None:
            self.numerical_columns = list(set(numerical_columns))
        if binary_columns is not None:
            self.binary_columns = list(set(binary_columns))

        # check if column exists overlap
        col_no_overlap, duplicate_cols = self._check_column_overlap(self.categorical_columns, self.numerical_columns,
                                                                    self.binary_columns)
        if not self.ignore_duplicate_cols:
            for col in duplicate_cols:
                logger.error(
                    f'Find duplicate cols named `{col}`, '
                    f'please process the raw data or set `ignore_duplicate_cols` to True!')
            assert col_no_overlap, ('The assigned categorical_columns, numerical_columns, '
                                    'binary_columns should not have overlap! Please check your input.')
        else:
            self._solve_duplicate_cols(duplicate_cols)

    def __call__(self, x, shuffle=True) -> Dict:
        """
        Parameters
        ----------
        x: pd.DataFrame
            with column names and features.

        shuffle: bool
            if shuffle column order during the training.

        Returns
        -------
        encoded_inputs: a dict with {
                'x_num': tensor contains numerical features,
                'num_col_input_ids': tensor contains numerical column tokenized ids,
                'x_cat_input_ids': tensor contains categorical column + feature ids,
                'x_bin_input_ids': tensor contains binary column + feature ids,
            }
        """
        encoded_inputs = {
            'x_num': None,
            'num_col_input_ids': None,
            'x_cat_input_ids': None,
            'x_bin_input_ids': None,
        }
        col_names = x.columns.tolist()
        # print('col_names', len(col_names))
        cat_cols = [c for c in col_names if
                    c in self.categorical_columns] if self.categorical_columns is not None else []
        num_cols = [c for c in col_names if c in self.numerical_columns] if self.numerical_columns is not None else []
        bin_cols = [c for c in col_names if c in self.binary_columns] if self.binary_columns is not None else []
        # print('cat_cols:', cat_cols)
        # print('num_cols:', num_cols)

        if len(cat_cols + num_cols + bin_cols) == 0:
            # take all columns as categorical columns!
            cat_cols = col_names

        if shuffle:
            np.random.shuffle(cat_cols)
            np.random.shuffle(num_cols)
            np.random.shuffle(bin_cols)

        # TODO:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # mask out NaN values like done in binary columns
        if len(num_cols) > 0:
            # print('num_cols', len(num_cols))
            # print(x.shape)
            x_num = x[num_cols]
            # print('x_num', x_num.shape)
            x_num = x_num.fillna(0)  # fill Nan with zero
            x_num = x_num.astype(float)
            x_num_ts = torch.tensor(x_num.values, dtype=float)   # x_num 转为tensor
            # print(x_num_ts.shape)
            num_col_ts = self.tokenizer(num_cols, padding=True, truncation=True, add_special_tokens=False,
                                        return_tensors='pt')

            '''
            print(num_cols)
             ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age', 'existing_credits', 
            'num_dependents']
            print(num_col_ts)
            {'input_ids': tensor([[9367, 0, 0, 0, 0],
                                  [4923, 1035, 3815, 0, 0],
                                  [18932, 1035, 8426, 0, 0],
                                  [5039, 1035, 2144, 0, 0],
                                  [2287, 0, 0, 0, 0],
                                  [4493, 1035, 6495, 0, 0],
                                  [16371, 2213, 1035, 7790, 2015]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0],
                                                                                               [0, 0, 0, 0, 0]]),
             'attention_mask': tensor([[1, 0, 0, 0, 0],
                                       [1, 1, 1, 0, 0],
                                       [1, 1, 1, 0, 0],
                                       [1, 1, 1, 0, 0],
                                       [1, 0, 0, 0, 0],
                                       [1, 1, 1, 0, 0],
                                       [1, 1, 1, 1, 1]])}
            '''
            encoded_inputs['x_num'] = x_num_ts
            encoded_inputs['num_col_input_ids'] = num_col_ts['input_ids']
            encoded_inputs['num_att_mask'] = num_col_ts['attention_mask']  # mask out attention
            # encoded_inputs = {
            #         'x_num': x_num_ts.to(device),  # x_num_ts 是 torch.Tensor
            #         'num_col_input_ids': num_col_ts['input_ids'].to(device),
            #         'num_att_mask': num_col_ts['attention_mask'].to(device)
            #     }

        if len(cat_cols) > 0:
            x_cat = x[cat_cols].astype(str)
            x_mask = (~pd.isna(x_cat)).astype(int)
            x_cat = x_cat.fillna('')
            x_cat = x_cat.apply(lambda x: x.name + ' ' + x) * x_mask  # mask out nan features
            x_cat_str = x_cat.agg(' '.join, axis=1).values.tolist()
            x_cat_ts = self.tokenizer(x_cat_str, padding=True, truncation=True, add_special_tokens=False,
                                      return_tensors='pt')
            encoded_inputs['x_cat_input_ids'] = x_cat_ts['input_ids']
            encoded_inputs['cat_att_mask'] = x_cat_ts['attention_mask']

        if len(bin_cols) > 0:
            x_bin = x[bin_cols]  # x_bin should already be integral (binary values in 0 & 1)
            x_bin_str = x_bin.apply(lambda x: x.name + ' ') * x_bin
            x_bin_str = x_bin_str.agg(' '.join, axis=1).values.tolist()
            x_bin_ts = self.tokenizer(x_bin_str, padding=True, truncation=True, add_special_tokens=False,
                                      return_tensors='pt')
            if x_bin_ts['input_ids'].shape[1] > 0:  # not all false
                encoded_inputs['x_bin_input_ids'] = x_bin_ts['input_ids']
                encoded_inputs['bin_att_mask'] = x_bin_ts['attention_mask']

        return encoded_inputs

    def save(self, path):
        """
        save the feature extractor configuration to local dir.
        """
        save_path = os.path.join(path, constants.EXTRACTOR_STATE_DIR)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save tokenizer
        tokenizer_path = os.path.join(save_path, constants.TOKENIZER_DIR)
        self.tokenizer.save_pretrained(tokenizer_path)

        # save other configurations
        coltype_path = os.path.join(save_path, constants.EXTRACTOR_STATE_NAME)
        col_type_dict = {
            'categorical': self.categorical_columns,
            'binary': self.binary_columns,
            'numerical': self.numerical_columns,
        }
        with open(coltype_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(col_type_dict))

    def load(self, path):
        """
        load the feature extractor configuration from local dir.
        """
        tokenizer_path = os.path.join(path, constants.TOKENIZER_DIR)
        coltype_path = os.path.join(path, constants.EXTRACTOR_STATE_NAME)

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        with open(coltype_path, 'r', encoding='utf-8') as f:
            col_type_dict = json.loads(f.read())

        self.categorical_columns = col_type_dict['categorical']
        self.numerical_columns = col_type_dict['numerical']
        self.binary_columns = col_type_dict['binary']
        logger.info(f'load feature extractor from {coltype_path}')

    def update(self, cat=None, num=None, bin=None):
        """
        update cat/num/bin column maps.
        """
        if cat is not None:
            self.categorical_columns.extend(cat)
            self.categorical_columns = list(set(self.categorical_columns))

        if num is not None:
            self.numerical_columns.extend(num)
            self.numerical_columns = list(set(self.numerical_columns))

        if bin is not None:
            self.binary_columns.extend(bin)
            self.binary_columns = list(set(self.binary_columns))

        col_no_overlap, duplicate_cols = self._check_column_overlap(self.categorical_columns, self.numerical_columns,
                                                                    self.binary_columns)
        if not self.ignore_duplicate_cols:
            for col in duplicate_cols:
                logger.error(
                    f'Find duplicate cols named `{col}`, '
                    f'please process the raw data or set `ignore_duplicate_cols` to True!')
            assert col_no_overlap, ('The assigned categorical_columns, numerical_columns, '
                                    'binary_columns should not have overlap! Please check your input.')
        else:
            self._solve_duplicate_cols(duplicate_cols)

    def _check_column_overlap(self, cat_cols=None, num_cols=None, bin_cols=None):
        all_cols = []
        if cat_cols is not None:
            all_cols.extend(cat_cols)
        if num_cols is not None:
            all_cols.extend(num_cols)
        if bin_cols is not None:
            all_cols.extend(bin_cols)
        org_length = len(all_cols)
        if org_length == 0:
            logger.warning(
                'No cat/num/bin cols specified, will take ALL columns as categorical! '
                'Ignore this warning if you specify the `checkpoint` to load the model.')
            return True, []
        unq_length = len(list(set(all_cols)))
        duplicate_cols = [item for item, count in collections.Counter(all_cols).items() if count > 1]
        return org_length == unq_length, duplicate_cols

    def _solve_duplicate_cols(self, duplicate_cols):
        for col in duplicate_cols:
            logger.warning('Find duplicate cols named `{col}`, will ignore it during training!')
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                self.categorical_columns.append(f'[cat]{col}')  # Why append?
            if col in self.numerical_columns:
                self.numerical_columns.remove(col)
                self.numerical_columns.append(f'[num]{col}')
            if col in self.binary_columns:
                self.binary_columns.remove(col)
                self.binary_columns.append(f'[bin]{col}')


class TabIDSFeatureProcessor(nn.Module):
    r"""
    Process inputs from feature extractor to map them to embeddings.
    """

    def __init__(self,
                 vocab_size=None,
                 hidden_dim=128,
                 hidden_dropout_prob=0,
                 pad_token_id=0,
                 device='cuda',
                 ) -> None:
        """
        args:
        categorical_columns: a list of categories feature names
        numerical_columns: a list of numerical feature names
        binary_columns: a list of yes or no feature names, accept binary indicators like
            (yes,no); (true,false); (0,1).
        """
        super().__init__()
        self.word_embedding = TabIDSWordEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            padding_idx=pad_token_id
        )
        self.num_embedding = TabIDSNumEmbedding(hidden_dim)
        self.align_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.device = device

    def _avg_embedding_by_mask(self, embs, att_mask=None):
        # Calculates the average embedding of tokens belonging to each column name
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if att_mask is None:
            return embs.mean(1)
        else:
            embs[att_mask == 0] = 0
            embs = embs.sum(1) / att_mask.sum(1, keepdim=True).to(device)  # to(embs.device)
            return embs

    def forward(self,
                x_num=None,
                num_col_input_ids=None,
                num_att_mask=None,
                x_cat_input_ids=None,
                cat_att_mask=None,
                x_bin_input_ids=None,
                bin_att_mask=None,
                **kwargs,
                ) -> Dict:
        """
        args:
        x: pd.DataFrame with column names and features.
        shuffle: if shuffle column order during the training.
        num_mask: indicate the NaN place of numerical features, 0: NaN 1: normal.
        """
        num_feat_embedding = None
        cat_feat_embedding = None
        bin_feat_embedding = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if x_num is not None and num_col_input_ids is not None:
            num_col_emb = self.word_embedding(
                num_col_input_ids.to(device))  # number of cat col, num of tokens, embedding size  # to(self.device))
            x_num = x_num.to(device)   # to(self.device))
            num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)
            # print("num_col_emb:",len(num_col_emb),"num_att_mask:",len(num_att_mask))
            num_feat_embedding = self.num_embedding(num_col_emb, x_num)        # 
            num_feat_embedding = self.align_layer(num_feat_embedding)     

        if x_cat_input_ids is not None:
            cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(device))  # to(self.device))
            cat_feat_embedding = self.align_layer(cat_feat_embedding)

        if x_bin_input_ids is not None:
            if x_bin_input_ids.shape[1] == 0:  # all false, pad zero
                x_bin_input_ids = torch.zeros(x_bin_input_ids.shape[0], dtype=int)[:, None]
            bin_feat_embedding = self.word_embedding(x_bin_input_ids.to(device))   # to(self.device))
            bin_feat_embedding = self.align_layer(bin_feat_embedding)

        # concat all embeddings
        emb_list = []
        att_mask_list = []
        if num_feat_embedding is not None:
            emb_list += [num_feat_embedding]
            att_mask_list += [torch.ones(num_feat_embedding.shape[0], num_feat_embedding.shape[1])]
        if cat_feat_embedding is not None:
            emb_list += [cat_feat_embedding]
            att_mask_list += [cat_att_mask]
        if bin_feat_embedding is not None:
            emb_list += [bin_feat_embedding]
            att_mask_list += [bin_att_mask]
        if len(emb_list) == 0:
            raise Exception('no feature found belonging into numerical, categorical, or binary, check your data!')
        all_feat_embedding = torch.cat(emb_list, 1).float()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        attention_mask = torch.cat(att_mask_list, 1).to(device)   # to(all_feat_embedding.device)
        return {'embedding': all_feat_embedding, 'attention_mask': attention_mask}


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == 'selu':
        return F.selu
    elif activation == 'leakyrelu':
        return F.leaky_relu
    raise RuntimeError("activation should be relu/gelu/selu/leakyrelu, not {}".format(activation))


class TabIDSTransformerLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,    # dropout = 0.1
                 layer_norm_eps=1e-5, batch_first=True, norm_first=False,
                 device='cuda', dtype=None, use_layer_norm=True) -> None:      #  device = None
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first,
                                               **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        # Implementation of gates
        self.gate_linear = nn.Linear(d_model, 1, bias=False)
        self.gate_act = nn.Sigmoid()

        self.norm_first = norm_first
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        key_padding_mask = ~key_padding_mask.bool()
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask
                        #    need_weights=True,
                        #    average_attn_weights=False
                           )[0]
        return self.dropout1(x)  #, attn_weights

    # feed forward block
    # def _ff_block(self, x: Tensor) -> Tensor:
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_act(self.gate_linear(x))  # [bs, num of tokens, 1]
        h = self.linear1(x)  # [bs, num of tokens, dim_feedforward]
        h = h * g  # add gate
        h = self.linear2(self.dropout(self.activation(h)))
        return self.dropout2(h)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None, **kwargs) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)

        if self.use_layer_norm:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(x + self._ff_block(x))

        else:  # do not use layer norm
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            x = x + self._ff_block(x)
        return x


class TabIDSInputEncoder(nn.Module):
    """
    Build a feature encoder that maps inputs tabular samples to embeddings.

    Parameters:
    -----------
    categorical_columns: list
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).

    ignore_duplicate_cols: bool
        if there is one column assigned to more than one type, e.g., the feature age is both nominated
        as categorical and binary columns, the model will raise errors. set True to avoid this error as
        the model will ignore this duplicate feature.

    disable_tokenizer_parallel: bool
        if the returned feature extractor is leveraged by the collate function for a dataloader,
        try to set this False in case the dataloader raises errors because the dataloader builds
        multiple workers and the tokenizer builds multiple workers at the same time.

    hidden_dim: int
        the dimension of hidden embeddings.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.

    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.

    """

    def __init__(self,
                 feature_extractor,
                 feature_processor,
                 device='cuda:0',
                 ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_processor = feature_processor
        self.device = device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, x):
        """
        Encode input tabular samples into embeddings.

        Parameters
        ----------
        x: pd.DataFrame
            with column names and features.
        """
        tokenized = self.feature_extractor(x)
        embeds = self.feature_processor(**tokenized)
        return embeds

    def load(self, ckpt_dir):
        # load feature extractor
        self.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))

        # load embedding layer
        model_name = os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'missing keys: {missing_keys}')
        logger.info(f'unexpected keys: {unexpected_keys}')
        logger.info(f'load model from {ckpt_dir}')


class TabIDSEncoder(nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 num_layer=4,
                 num_attention_head=4,
                 hidden_dropout_prob=0,
                 ffn_dim=256,
                 activation='relu',
                 ):
        super().__init__()
        self.transformer_encoder = nn.ModuleList(
            [
                TabIDSTransformerLayer(
                    d_model=hidden_dim,
                    nhead=num_attention_head,
                    dropout=hidden_dropout_prob,
                    dim_feedforward=ffn_dim,
                    batch_first=True,
                    layer_norm_eps=1e-5,
                    norm_first=False,
                    use_layer_norm=True,
                    activation=activation, )
            ]
        )
        if num_layer > 1:
            encoder_layer = TabIDSTransformerLayer(d_model=hidden_dim,
                                                     nhead=num_attention_head,
                                                     dropout=hidden_dropout_prob,
                                                     dim_feedforward=ffn_dim,
                                                     batch_first=True,
                                                     layer_norm_eps=1e-5,
                                                     norm_first=False,
                                                     use_layer_norm=True,
                                                     activation=activation,
                                                     )
            stacked_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layer - 1)
            self.transformer_encoder.append(stacked_transformer)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        """
        args:
        embedding: bs, num_token, hidden_dim
        """
        outputs = embedding
        for i, mod in enumerate(self.transformer_encoder):
            outputs = mod(outputs, src_key_padding_mask=attention_mask)
        return outputs


class TabIDSLinearClassifier(nn.Module):
    def __init__(self,
                 num_class,
                 hidden_dim=128) -> None:
        super().__init__()
        if num_class <= 2:
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            self.fc = nn.Linear(hidden_dim, num_class)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device) 

        x = x[:, 0, :]  # take the cls token embedding
        x = self.norm(x)
        logits = self.fc(x)
        return logits


class TabIDSProjectionHead(nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 projection_dim=128):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, projection_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h = h.to(device)
        h = self.dense(x)
        return h


class TabIDSCLSToken(nn.Module):
    """
    add a learnable cls token embedding at the end of each sequence.
    """

    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.weight, a=-1 / math.sqrt(hidden_dim), b=1 / math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def expand(self, *leading_dimensions):
        new_dims = (1,) * (len(leading_dimensions) - 1)  # (1,)
        # self.weight.shape: [hidden_dim]
        # self.weight.view(*new_dims, -1).shape: [1, hidden_dim]
        # self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1).shape: [bs, 1, hidden_dim]
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, embedding, attention_mask=None, **kwargs) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding = embedding.to(device)
        embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1)
        outputs = {'embedding': embedding}
        if attention_mask is not None:
            attention_mask = torch.cat(
                [torch.ones(attention_mask.shape[0], 1).to(device), attention_mask], 1)  # .to(attention_mask.device)
        outputs['attention_mask'] = attention_mask
        return outputs


class TabIDSModel(nn.Module):
    """
    The base transtab model for downstream tasks like contrastive learning, binary classification, etc.
    All models subclass this basemodel and usually rewrite the ``forward`` function. Refer to the source code of
    :class:`transtab.modeling_transtab.TransTabClassifier` or :class:`transtab.modeling_transtab.TransTabForCL`
    for the implementation details.

    Parameters
    ----------
    categorical_columns: list
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).

    feature_extractor: TransTabFeatureExtractor
        a feature extractor to tokenize the input tables. if not passed the model will build itself.

    hidden_dim: int
        the dimension of hidden embeddings.

    num_layer: int
        the number of transformer layers used in the encoder.

    num_attention_head: int
        the numebr of heads of multihead self-attention layer in the transformers.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.

    ffn_dim: int
        the dimension of feed-forward layer in the transformer layer.

    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.

    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.

    Returns
    -------
    A TransTabModel model.
    """

    def __init__(self,
                 categorical_columns=None,
                 numerical_columns=None,
                 binary_columns=None,
                 feature_extractor=None,
                 hidden_dim=128,
                 num_layer=2,
                 num_attention_head=8,
                 hidden_dropout_prob=0.1,   # 0.1
                 ffn_dim=256,
                 activation='relu',
                 device='cuda',
                 **kwargs,
                 ) -> None:

        super().__init__()
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns
        if categorical_columns is not None:
            self.categorical_columns = list(set(categorical_columns))
        if numerical_columns is not None:
            self.numerical_columns = list(set(numerical_columns))
        if binary_columns is not None:
            self.binary_columns = list(set(binary_columns))

        if feature_extractor is None:
            feature_extractor = TabIDSFeatureExtractor(
                categorical_columns=self.categorical_columns,
                numerical_columns=self.numerical_columns,
                binary_columns=self.binary_columns,
                **kwargs,
            )

        feature_processor = TabIDSFeatureProcessor(
            vocab_size=feature_extractor.vocab_size,
            pad_token_id=feature_extractor.pad_token_id,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            device=device,
        )

        self.input_encoder = TabIDSInputEncoder(
            feature_extractor=feature_extractor,
            feature_processor=feature_processor,
            device=device,
        )

        self.encoder = TabIDSEncoder(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
        )

        self.cls_token = TabIDSCLSToken(hidden_dim=hidden_dim)
        self.device = device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, x, y=None):
        """Extract the embeddings based on input tables.

        Parameters
        ----------
        x: pd.DataFrame
            a batch of samples stored in pd.DataFrame.

        y: pd.Series
            the corresponding labels for each sample in ``x``. ignored for the basemodel.

        Returns
        -------
        final_cls_embedding: torch.Tensor
            the [CLS] embedding at the end of transformer encoder.
        """
        embeded = self.input_encoder(x)
        embeded = self.cls_token(**embeded)

        # go through transformers, get final cls embedding
        encoder_output = self.encoder(**embeded)

        # get cls token
        final_cls_embedding = encoder_output[:, 0, :]
        return final_cls_embedding

    def load(self, ckpt_dir):
        '''Load the model state_dict and feature_extractor configuration
        from the ``ckpt_dir``.

        Parameters
        ----------
        ckpt_dir: str
            the directory path to load.

        Returns
        -------
        None

        '''
        # load model weight state dict
        model_name = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'missing keys: {missing_keys}')
        logger.info(f'unexpected keys: {unexpected_keys}')
        logger.info(f'load model from {ckpt_dir}')

        # load feature extractor
        self.input_encoder.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

    def save(self, ckpt_dir):
        '''Save the model state_dict and feature_extractor configuration
        to the ``ckpt_dir``.

        Parameters
        ----------
        ckpt_dir: str
            the directory path to save.

        Returns
        -------
        None

        '''
        # save model weight state dict
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(ckpt_dir, constants.WEIGHTS_NAME))
        if self.input_encoder.feature_extractor is not None:
            self.input_encoder.feature_extractor.save(ckpt_dir)

        # save the input encoder separately
        state_dict_input_encoder = self.input_encoder.state_dict()
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME))
        return None

    def update(self, config):
        """
        Update the configuration of feature extractor's column map for cat, num, and bin cols.
        Or update the number of classes for the output classifier layer.

        Parameters
        ----------
        config: dict
            a dict of configurations: keys cat:list, num:list, bin:list are to specify the new column names;
            key num_class:int is to specify the number of classes for finetuning on a new dataset.

        Returns
        -------
        None

        """

        col_map = {}
        for k, v in config.items():
            if k in ['cat', 'num', 'bin']:
                col_map[k] = v

        self.input_encoder.feature_extractor.update(**col_map)
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

        if 'num_class' in config:
            num_class = config['num_class']
            self._adapt_to_new_num_class(num_class)

        return None

    def _check_column_overlap(self, cat_cols=None, num_cols=None, bin_cols=None):
        all_cols = []
        if cat_cols is not None: all_cols.extend(cat_cols)
        if num_cols is not None: all_cols.extend(num_cols)
        if bin_cols is not None: all_cols.extend(bin_cols)
        org_length = len(all_cols)
        unq_length = len(list(set(all_cols)))
        duplicate_cols = [item for item, count in collections.Counter(all_cols).items() if count > 1]
        return org_length == unq_length, duplicate_cols

    def _solve_duplicate_cols(self, duplicate_cols):
        for col in duplicate_cols:
            logger.warning('Find duplicate cols named `{col}`, will ignore it during training!')
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                self.categorical_columns.append(f'[cat]{col}')
            if col in self.numerical_columns:
                self.numerical_columns.remove(col)
                self.numerical_columns.append(f'[num]{col}')
            if col in self.binary_columns:
                self.binary_columns.remove(col)
                self.binary_columns.append(f'[bin]{col}')

    def _adapt_to_new_num_class(self, num_class):
        if num_class != self.num_class:
            self.num_class = num_class
            self.clf = TabIDSLinearClassifier(num_class, hidden_dim=self.cls_token.hidden_dim)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.clf.to(self.device)
            self.clf.to(device)
            print('num_class:', num_class)
            if self.num_class > 2:
                self.loss_fn = nn.CrossEntropyLoss(reduction='none')
            else:
                print('num_class:', num_class)
                self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            logger.info(f'Build a new classifier with num {num_class} classes outputs, need further finetune to work.')


class TabIDS(TabIDSModel):
    """
    The classifier model subclass from :class:`transtab.modeling_transtab.TransTabModel`.

    Parameters
    ----------
    categorical_columns: list
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).

    feature_extractor: TransTabFeatureExtractor
        a feature extractor to tokenize the input tables. if not passed the model will build itself.

    num_class: int
        number of output classes to be predicted.

    hidden_dim: int
        the dimension of hidden embeddings.

    num_layer: int
        the number of transformer layers used in the encoder.

    num_attention_head: int
        the numebr of heads of multihead self-attention layer in the transformers.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.

    ffn_dim: int
        the dimension of feed-forward layer in the transformer layer.

    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.

    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.

    Returns
    -------
    A TransTabClassifier model.

    """

    def __init__(self,
                 categorical_columns=None,
                 numerical_columns=None,
                 binary_columns=None,
                 feature_extractor=None,
                 num_class=2,
                 hidden_dim=128,
                 num_layer=2,
                 num_attention_head=8,
                 hidden_dropout_prob=0,
                 ffn_dim=256,
                 activation='relu',
                 device='cuda',
                 **kwargs,
                 ) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            device=device,
            **kwargs,
        )
        self.num_class = num_class
        self.categorical_columns = categorical_columns
        

        self.clf = TabIDSLinearClassifier(num_class=num_class, hidden_dim=hidden_dim)
        if self.num_class > 2:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, x, y=None):
        """
        Make forward pass given the input feature ``x`` and label ``y`` (optional).

        Parameters
        ----------
        x: pd.DataFrame or dict
            pd.DataFrame: a batch of raw tabular samples; dict: the output of TransTabFeatureExtractor.

        y: pd.Series
            the corresponding labels for each sample in ``x``. if label is given, the model will return
            the classification loss by ``self.loss_fn``.

        Returns
        -------
        logits: torch.Tensor
            the [CLS] embedding at the end of transformer encoder.

        loss: torch.Tensor or None
            the classification loss.

        """
        if isinstance(x, dict):
            # input is the pre-tokenized encoded inputs
            inputs = x
        elif isinstance(x, pd.DataFrame):
            # input is dataframe
            # print('Classifier:', x.shape)
            inputs = self.input_encoder.feature_extractor(x)
        else:
            raise ValueError(f'TransTabClassifier takes inputs with dict or pd.DataFrame, find {type(x)}.')

        # print("x[x_num].shape",x['x_num'].shape)
        # print("inputs.keys()",inputs.keys())
        # print('inputs[x_num].shape',inputs['x_num'].shape)

        outputs = self.input_encoder.feature_processor(**inputs)
        outputs = self.cls_token(**outputs)

        # go through transformers, get the first cls embedding
        encoder_output = self.encoder(**outputs)  # bs, seqlen+1, hidden_dim

        # classifier
        logits = self.clf(encoder_output)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if y is not None:
            # compute classification loss
            if self.num_class == 2:             # ?二分类 
                y_ts = torch.tensor(y.values.astype(np.float32)).to(device)
                loss = self.loss_fn(logits.flatten(), y_ts)
            else:
                y_ts = torch.tensor(y.values).to(device).long()
                loss = self.loss_fn(logits, y_ts)
            loss = loss.mean()
        else:
            loss = None

        return logits, loss


