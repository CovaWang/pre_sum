import os
import re
import gc
import copy
import csv
import json
import logging
import torch
from tqdm import tqdm
from time import time
from multiprocessing import Pool
from functools import partial
import torch
from torch.utils.data import TensorDataset, IterableDataset

logger = logging.getLogger(__name__)


class FSTensorDataset(IterableDataset):
    def __init__(self, files_list):
        super(FSTensorDataset).__init__()
        self.files_list = files_list

    def __iter__(self):
        for data_file in self.files_list:
            dataset_section = torch.load(data_file)
            for example in dataset_section:
                yield example
            # 在加载下一个文件之前清除内存使用情况
            dataset_section = None
            gc.collect()
            del dataset_section
            gc.collect()


class InputExample(object):
    """
    简单序列分类的单个训练/测试示例。

    参数:
        guid: 示例的唯一标识。
        text_a: 字符串。第一个序列的未分词文本。对于单序列任务，只需要指定这个序列。
        text_b: (可选) 字符串。第二个序列的未分词文本。只有对于序列对任务才需要指定。
        label: (可选) 字符串。示例的标签。应该为训练和开发示例指定，但不为测试示例指定。
    """

    def __init__(self, guid, text, labels):
        self.guid = guid
        self.text = text
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """将此实例序列化为Python字典。"""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """将此实例序列化为JSON字符串。"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    数据的单个特征集。

    参数:
        attention_mask: 避免对填充标记索引执行注意力的掩码。
            掩码值选择在``[0, 1]``中：
            通常``1``表示未被掩盖的标记，``0``表示被掩盖（填充）标记。
        token_type_ids: 段标记索引以指示输入的第一部分和第二部分。
        label: 与输入对应的标签
    """

    def __init__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        sent_rep_token_ids=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels
        self.sent_rep_token_ids = sent_rep_token_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """将此实例序列化为Python字典。"""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """将此实例序列化为JSON字符串。"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SentencesProcessor:
    def __init__(self, name=None, labels=None, examples=None, verbose=False):
        self.name = name
        self.labels = [] if labels is None else labels
        self.examples = [] if examples is None else examples
        self.verbose = verbose

    def __len__(self):
        return len(self.examples)

    @classmethod
    def create_from_csv(
        cls,
        file_name,
        split_name="",
        column_label=0,
        column_text=1,
        column_id=None,
        skip_first_row=False,
        **kwargs
    ):
        processor = cls(**kwargs)
        processor.add_examples_from_csv(
            file_name,
            split_name=split_name,
            column_label=column_label,
            column_text=column_text,
            column_id=column_id,
            skip_first_row=skip_first_row,
            overwrite_labels=True,
            overwrite_examples=True,
        )
        return processor

    @classmethod
    def create_from_examples(cls, texts_or_text_and_labels, labels=None, **kwargs):
        processor = cls(**kwargs)
        processor.add_examples(texts_or_text_and_labels, labels=labels)
        return processor

    def add_examples_from_csv(
        self,
        file_name,
        split_name="",
        column_label=0,
        column_text=1,
        column_id=None,
        skip_first_row=False,
        overwrite_labels=False,
        overwrite_examples=False,
    ):
        lines = self._read_tsv(file_name)
        if skip_first_row:
            lines = lines[1:]
        texts = []
        labels = []
        ids = []
        for (i, line) in enumerate(lines):
            texts.append(line[column_text])
            labels.append(line[column_label])
            if column_id is not None:
                ids.append(line[column_id])
            else:
                guid = "%s-%s" % (split_name, i) if split_name else "%s" % i
                ids.append(guid)

        return self.add_examples(
            texts,
            labels,
            ids,
            overwrite_labels=overwrite_labels,
            overwrite_examples=overwrite_examples,
        )

    def add_examples(
        self,
        texts,
        labels=None,
        ids=None,
        oracle_ids=None,
        overwrite_labels=False,
        overwrite_examples=False,
    ):
        assert texts  # 不是空数组
        assert labels is None or len(texts) == len(labels)
        assert ids is None or len(texts) == len(ids)
        assert not (labels and oracle_ids)
        assert isinstance(texts, list)

        if ids is None:
            ids = [None] * len(texts)
        if labels is None:
            if oracle_ids:  # 将`oracle_ids`转换为`labels`
                labels = []
                for text_set, oracle_id in zip(texts, oracle_ids):
                    text_label = [0] * len(text_set)
                    for l in oracle_id:
                        text_label[l] = 1
                    labels.append(text_label)
            else:
                labels = [None] * len(texts)

        examples = []
        added_labels = list()
        for (text_set, label_set, guid) in zip(texts, labels, ids):
            if not text_set or not label_set:
                continue  # input()
            added_labels.append(label_set)
            examples.append(InputExample(guid=guid, text=text_set, labels=label_set))

        # 更新示例
        if overwrite_examples:
            self.examples = examples
        else:
            self.examples.extend(examples)

        # 更新标签
        if overwrite_labels:
            self.labels = added_labels
        else:
            self.labels += added_labels

        return self.examples

    # def preprocess_examples(self, examples, labels, min_sentence_ntokens=5, max_sentence_ntokens=200, min_example_nsents=3, max_example_nsents=100):
    #     for (ex_index, example) in enumerate(examples):
    #         if ex_index % 10000 == 0:
    #             logger.info("预处理示例 %d", ex_index)
    #         # 如果它们大于`min_sentence_ntokens`，则选择`example.text`中的句子索引
    #         idxs = [i for i, s in enumerate(example.text) if (len(s) > min_sentence_ntokens)]
    #         # 将选定的源句子截断为`max_sentence_ntokens`
    #         example.text = [example.text[i][:max_sentence_ntokens] for i in idxs]
    #         # 仅选择与长度要求匹配的句子的标签
    #         example.labels = [example.labels[i] for i in idxs]
    #         # 将整个源截断为最大句子数（`max_example_nsents`）
    #         example.text = example.text[:max_example_nsents]
    #         # 对标签执行上述截断
    #         example.labels = example.labels[:max_example_nsents]

    #         # 如果示例不符合长度要求，则将其删除
    #         if (len(example.text) < min_example_nsents):
    #             examples.pop(ex_index)
    #             labels.pop(ex_index)
    #         return examples, labels

    def get_features_process(
        self,
        features,
        input_information,
        num_examples=0,
        tokenizer=None,
        bert_compatible_cls=True,
        create_sent_rep_token_ids=True,
        sent_rep_token_id=None,
        create_segment_ids="binary",
        segment_token_id=None,
        max_length=None,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
    ):
        ex_index, example, label = input_information
        if ex_index % 1000 == 0:
            logger.info(
                "为示例生成特征 "
                + str(ex_index)
                + "/"
                + str(num_examples)
            )
        if (
            bert_compatible_cls
        ):  # 在每个句子之间添加一个'[CLS]'标记并输出`input_ids`
            # 将`example.text`转换为句子数组
            src_txt = [" ".join(sent) for sent in example.text]
            # 用' [SEP] [CLS] '分隔每个句子并转换为字符串
            text = " [SEP] [CLS] ".join(src_txt)
            # 分词
            src_subtokens = tokenizer.tokenize(text)
            # 选择前`(max_length-2)`个标记（以便可以添加后续标记行）
            src_subtokens = src_subtokens[: (max_length - 2)]
            # 在开头添加'[CLS]'并在结尾添加'[SEP]'
            src_subtokens = ["[CLS]"] + src_subtokens + ["[SEP]"]
            # 创建`input_ids`
            input_ids = tokenizer.convert_tokens_to_ids(src_subtokens)
        else:
            input_ids = tokenizer.encode(
                example.text,
                add_special_tokens=True,
                max_length=min(max_length, tokenizer.max_len),
            )

        # 段（标记类型）ID
        segment_ids = None
        if create_segment_ids == "binary":
            current_segment_flag = True
            segment_ids = []
            for token in input_ids:
                if token == segment_token_id:
                    current_segment_flag = not current_segment_flag
                segment_ids += [0 if current_segment_flag else 1]

        if create_segment_ids == "sequential":
            current_segment = 0
            segment_ids = []
            for token in input_ids:
                if token == segment_token_id:
                    current_segment += 1
                segment_ids += [current_segment]

        # 句子表示标记ID
        sent_rep_ids = None
        if create_sent_rep_token_ids:
            # 创建`sent_rep`标记的索引列表
            sent_rep_ids = [
                i for i, t in enumerate(input_ids) if t == sent_rep_token_id
            ]
            # 将`label`截断为`cls_ids`的长度，即句子的数量
            label = label[: len(sent_rep_ids)]

        # 注意
        # 掩码对真实标记为1，对填充标记为0。仅对真实
        # 标记进行注意。
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # 填充
        # 零填充到序列长度。
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + attention_mask
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + (
                [0 if mask_padding_with_zero else 1] * padding_length
            )

        assert len(input_ids) == max_length, "输入长度错误 {} vs {}".format(
            len(input_ids), max_length
        )
        assert (
            len(attention_mask) == max_length
        ), "输入长度错误 {} vs {}".format(len(attention_mask), max_length)

        if ex_index < 5 and self.verbose:
            logger.info("*** 示例 ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            if segment_ids is not None:
                logger.info(
                    "token_type_ids: %s" % " ".join([str(x) for x in segment_ids])
                )
            if sent_rep_ids is not None:
                logger.info(
                    "sent_rep_token_ids: %s" % " ".join([str(x) for x in sent_rep_ids])
                )
            logger.info(
                "attention_mask: %s" % " ".join([str(x) for x in attention_mask])
            )
            logger.info("labels: %s (id = %s)" % (example.labels, label))

        # 返回特征
        return InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label,
            token_type_ids=segment_ids,
            sent_rep_token_ids=sent_rep_ids,
        )

    def get_features(
        self,
        tokenizer,
        bert_compatible_cls=True,
        create_sent_rep_token_ids=True,
        sent_rep_token_id=None,
        create_segment_ids="binary",
        segment_token_id=None,
        n_process=2,
        max_length=None,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
        return_tensors=False,
        save_to_path=None,
        save_to_name=None,
    ):
        """
        将示例转换为``InputFeatures``的列表

        参数:
            tokenizer: 将对示例进行分词的tokenizer实例
            max_length: 最大示例长度
            task: GLUE任务
            label_list: 标签列表。可以通过处理器使用``processor.get_labels()``方法获得
            output_mode: 指示输出模式的字符串。可以是``regression``或``classification``
            pad_on_left: 如果设置为``True``，则示例将在左侧而不是右侧填充（默认）
            pad_token: 填充标记
            mask_padding_with_zero: 如果设置为``True``，则注意力掩码将用``1``填充实际值
                并用``0``填充填充值。如果设置为``False``，则反转（``1``表示填充值，``0``表示
                实际值）

        返回:
            如果``examples``输入是``tf.data.Dataset``，将返回一个``tf.data.Dataset``
            包含特定于任务的特征。如果输入是``InputExamples``的列表，将返回
            一个任务特定的``InputFeatures``列表，可以输入到模型中。

        """
        if max_length is None:
            max_length = tokenizer.max_len

        # batch_length = max(len(input_ids) for input_ids in all_input_ids)

        if create_sent_rep_token_ids:
            if sent_rep_token_id == "sep":  # 获取sep标记ID
                sent_rep_token_id = tokenizer.sep_token_id
            elif sent_rep_token_id == "cls":  # 获取cls标记ID
                sent_rep_token_id = tokenizer.cls_token_id
            else:  # 默认尝试获取`sep_token_id`，如果未设置`sent_rep_token_id`
                sent_rep_token_id = tokenizer.sep_token_id

        if create_segment_ids:
            if segment_token_id == "period":  # 获取“.”的标记ID
                segment_token_id = tokenizer.convert_tokens_to_ids(["."])[0]
            else:  # 默认尝试获取`cls_token_id`，如果未设置`segment_token_id`
                segment_token_id = tokenizer.cls_token_id

        features = []
        pool = Pool(n_process)
        _get_features_process = partial(
            self.get_features_process,
            features,
            num_examples=len(self.labels),
            tokenizer=tokenizer,
            bert_compatible_cls=bert_compatible_cls,
            create_sent_rep_token_ids=create_sent_rep_token_ids,
            sent_rep_token_id=sent_rep_token_id,
            create_segment_ids=create_segment_ids,
            segment_token_id=segment_token_id,
            max_length=max_length,
            pad_on_left=pad_on_left,
            pad_token=pad_token,
            mask_padding_with_zero=mask_padding_with_zero,
        )

        for rtn_features in pool.map(
            _get_features_process,
            zip(range(len(self.labels)), self.examples, self.labels),
        ):
            features.append(rtn_features)

        pool.close()
        pool.join()

        # for (ex_index, (example, label)) in tqdm(enumerate(zip(self.examples, self.labels)), total=len(self.labels), desc="Creating Features"):
        #     # if ex_index % 1000 == 0:
        #     #     logger.info("为示例分词 %d", ex_index)

        #     if bert_compatible_cls: # 在每个句子之间添加一个'[CLS]'标记并输出`input_ids`
        #         # 将`example.text`转换为句子数组
        #         src_txt = [' '.join(sent) for sent in example.text]
        #         # 用' [SEP] [CLS] '分隔每个句子并转换为字符串
        #         text = ' [SEP] [CLS] '.join(src_txt)
        #         # 分词
        #         src_subtokens = tokenizer.tokenize(text)
        #         # 选择前`(max_length-2)`个标记（以便可以添加后续标记行）
        #         src_subtokens = src_subtokens[:(max_length-2)]
        #         # 在开头添加'[CLS]'并在结尾添加'[SEP]'
        #         src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
        #         # 创建`input_ids`
        #         input_ids = tokenizer.convert_tokens_to_ids(src_subtokens)
        #     else:
        #         input_ids = tokenizer.encode(
        #             example.text, add_special_tokens=True, max_length=min(max_length, tokenizer.max_len),
        #         )

        #     # 段（标记类型）ID
        #     segment_ids = None
        #     if create_segment_ids == "binary":
        #         current_segment_flag = True
        #         segment_ids = []
        #         for token in input_ids:
        #             if token == segment_token_id:
        #                 current_segment_flag = not current_segment_flag
        #             segment_ids += [0 if current_segment_flag else 1]

        #     if create_segment_ids == "sequential":
        #         current_segment = 0
        #         segment_ids = []
        #         for token in input_ids:
        #             if token == segment_token_id:
        #                 current_segment += 1
        #             segment_ids += [current_segment]

        #     # 句子表示标记ID
        #     sent_rep_ids = None
        #     if create_sent_rep_token_ids:
        #         # 创建`sent_rep`标记的索引列表
        #         sent_rep_ids = [i for i, t in enumerate(input_ids) if t == sent_rep_token_id]
        #         # 将`label`截断为`cls_ids`的长度，即句子的数量
        #         label = label[:len(sent_rep_ids)]

        #     # 注意
        #     # 掩码对真实标记为1，对填充标记为0。仅对真实
        #     # 标记进行注意。
        #     attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        #     # 填充
        #     # 零填充到序列长度。
        #     padding_length = max_length - len(input_ids)
        #     if pad_on_left:
        #         input_ids = ([pad_token] * padding_length) + input_ids
        #         attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        #     else:
        #         input_ids = input_ids + ([pad_token] * padding_length)
        #         attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

        #     assert len(input_ids) == max_length, "输入长度错误 {} vs {}".format(
        #         len(input_ids), max_length
        #     )
        #     assert len(attention_mask) == max_length, "输入长度错误 {} vs {}".format(
        #         len(attention_mask), max_length
        #     )

        #     if ex_index < 5 and self.verbose:
        #         logger.info("*** 示例 ***")
        #         logger.info("guid: %s" % (example.guid))
        #         logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #         if segment_ids is not None:
        #             logger.info("token_type_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #         if sent_rep_ids is not None:
        #             logger.info("sent_rep_token_ids: %s" % " ".join([str(x) for x in sent_rep_ids]))
        #         logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #         logger.info("labels: %s (id = %s)" % (example.labels, label))

        #     # Append to features
        #     features.append(
        #         InputFeatures(
        #             input_ids=input_ids,
        #             attention_mask=attention_mask,
        #             labels=label,
        #             token_type_ids=segment_ids,
        #             sent_rep_token_ids=sent_rep_ids
        #         )
        #     )

        if return_tensors is False:
            return features
        else:

            def pad(data, pad_id, width=None):
                if not width:
                    width = max(len(d) for d in data)
                rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
                return rtn_data

            # Pad sentence representation token ids (`sent_rep_token_ids`)
            all_sent_rep_token_ids_padded = torch.tensor(
                pad([f.sent_rep_token_ids for f in features], -1), dtype=torch.long
            )
            all_sent_rep_token_ids_masks = ~(all_sent_rep_token_ids_padded == -1)
            all_sent_rep_token_ids_padded[all_sent_rep_token_ids_padded == -1] = 0

            all_input_ids = torch.tensor(
                [f.input_ids for f in features], dtype=torch.long
            )
            all_attention_masks = torch.tensor(
                [f.attention_mask for f in features], dtype=torch.long
            )
            all_labels = torch.tensor(
                pad([f.labels for f in features], 0), dtype=torch.long
            )
            all_token_type_ids = torch.tensor(
                pad([f.token_type_ids for f in features], 0), dtype=torch.long
            )
            all_sent_rep_token_ids = torch.tensor(
                all_sent_rep_token_ids_padded, dtype=torch.long
            )
            all_sent_rep_token_ids_masks = torch.tensor(
                all_sent_rep_token_ids_masks, dtype=torch.long
            )

            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_labels,
                all_token_type_ids,
                all_sent_rep_token_ids,
                all_sent_rep_token_ids_masks,
            )

            if save_to_path:
                final_save_name = (
                    save_to_name if save_to_name else ("dataset_" + self.name)
                )
                dataset_path = os.path.join(save_to_path, (final_save_name + ".pt"),)
                logger.info("Saving dataset into cached file %s", dataset_path)
                torch.save(dataset, dataset_path)

            return dataset

    def load(self, load_from_path, dataset_name=None):
        """ Attempts to load the dataset from storage. If that fails, will return None. """
        final_load_name = dataset_name if dataset_name else ("dataset_" + self.name)
        dataset_path = os.path.join(load_from_path, (final_load_name + ".pt"),)
        if os.path.exists(dataset_path):
            logger.info("Loading data from file %s", dataset_path)
            dataset = torch.load(dataset_path)
            return dataset
        else:
            return None