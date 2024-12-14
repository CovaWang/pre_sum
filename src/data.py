import os
import re
import gc
import copy
import csv
import json
import random
import logging
import torch
from tqdm import tqdm
from time import time
from multiprocessing import Pool
from functools import partial
import torch
import torch.nn.functional as F
from torch._six import container_abcs
from torch.utils.data import TensorDataset, IterableDataset

import numpy as np

logger = logging.getLogger(__name__)


def pad(data, pad_id, width=None, pad_on_left=False):
    """Pad `data` with `pad_id` to `width` on the right by default but if `pad_on_left` then left."""
    if not width:
        width = max(len(d) for d in data)
    if pad_on_left:
        rtn_data = [[pad_id] * (width - len(d)) + d for d in data]
    else:
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data


def pad_tensors(tensors, pad_id=0, width=None, pad_on_left=False):
    """Pad `tensors` with `pad_id` to `width` on the right by default but if `pad_on_left` then left."""
    if not width:
        width = max(len(d) for d in tensors)
    if pad_on_left:
        pad_params = ((width - len(tensor)), 0)
    else:
        pad_params = (0, (width - len(tensor)))
    return [
        F.pad(tensor, pad=pad_params, mode="constant", value=pad_id)
        for tensor in tensors
    ]


def pad_batch_collate(batch):
    """
    Collate function to be passed to `DataLoaders`.
    PyTorch Docs: https://pytorch.org/docs/stable/data.html#dataloader-collate-fn

    Calculates padding (per batch for efficiency) of `labels` and `token_type_ids`
    if they exist within the batch from the `Dataset`. Also, pads `sent_rep_token_ids`
    and creates the `sent_rep_mask` to indicate which numbers in the `sent_rep_token_ids`
    list are actually the locations of sentence representation ids and which are padding.
    Finally, calculates the `attention_mask` for each set of `input_ids` and pads both the
    `attention_mask` and the `input_ids`. Converts all inputs to tensors.

    If `sent_lengths` are found then they will also automatically be padded. However, the
    padding for sentence lengths is complicated. Each list of sentence lengths needs to be
    the length of the longest list of sentence lengths and the sum of all the lengths in each
    list needs to add to the length of the input_ids width (the length of each input_id). The
    second requirement exists because `torch.split()` (which is used in the `mean_tokens` pooling
    algorithm to convert word vectors to sentence embeddings in `pooling.py`) will split a
    tensor into the lengths requested but will error instead of returning any extra. However,
    `torch.split()` will split a tensor into zero length segments. Thus, to solve this, zeros
    are added to each sentence length list for each example until one more padding value is needed
    to get the maximum number of sentences. Once only one more value is needed, the total value
    needded to reach the width of the `input_ids` is added.

    `source` and `target`, if present, are simply passed on without any processing. Therefore,
    the standard `collate_fn` function for `DataLoader`s will not work if these are present since
    they cannot be converted to tensors without padding. This `collate_fun` must be used if 
    `source` or `target` is present in the loaded dataset. 
    """
    elem = batch[0]
    elem_type = type(elem)
    final_dictionary = {}

    for key in elem:
        if key == "sent_lengths":
            continue

        feature_list = [d[key] for d in batch]
        if key == "sent_rep_token_ids":
            feature_list = pad(feature_list, -1)
            sent_rep_token_ids = torch.tensor(feature_list)

            sent_rep_mask = ~(sent_rep_token_ids == -1)
            sent_rep_token_ids[sent_rep_token_ids == -1] = 0

            final_dictionary["sent_rep_token_ids"] = sent_rep_token_ids
            final_dictionary["sent_rep_mask"] = sent_rep_mask
            continue  # go to next key
        elif key == "input_ids":
            input_ids = feature_list

            # Attention
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [[1] * len(ids) for ids in input_ids]

            input_ids_width = max(len(ids) for ids in input_ids)
            input_ids = pad(input_ids, 0, width=input_ids_width)
            input_ids = torch.tensor(input_ids)
            attention_mask = pad(attention_mask, 0)
            attention_mask = torch.tensor(attention_mask)

            if "sent_lengths" in elem:
                sent_lengths = []
                sent_lengths_mask = []
                sent_lengths_width = max(len(d["sent_lengths"]) + 1 for d in batch)
                for d in batch:
                    current_sent_lens = d["sent_lengths"]
                    current_sent_lengths_mask = [True] * len(current_sent_lens)
                    num_to_add = sent_lengths_width - len(current_sent_lens)
                    total_value_to_add = input_ids_width - sum(current_sent_lens)
                    while num_to_add > 1:
                        num_to_add -= 1
                        # total_value_to_add -= 1
                        current_sent_lens.append(0)
                        current_sent_lengths_mask.append(False)
                    if total_value_to_add > 0:
                        current_sent_lens.append(total_value_to_add)
                        current_sent_lengths_mask.append(False)

                    sent_lengths.append(current_sent_lens)
                    sent_lengths_mask.append(current_sent_lengths_mask)
                final_dictionary["sent_lengths"] = sent_lengths
                final_dictionary["sent_lengths_mask"] = torch.tensor(sent_lengths_mask)

            final_dictionary["input_ids"] = input_ids
            final_dictionary["attention_mask"] = attention_mask

            continue

        elif key == "source" or key == "target":
            final_dictionary[key] = feature_list
            continue

        elif key == "labels" or key == "token_type_ids":
            feature_list = pad(feature_list, 0)

        feature_list = torch.tensor(feature_list)
        final_dictionary[key] = feature_list

    return final_dictionary


class FSIterableDataset(IterableDataset):
    """
    A dataset to yield examples from a list of files that are saved python objects that
    can be iterated over. These files could be other PyTorch datasets (tested with
    `TensorDataset`) or other python objects such as lists, for example. Each file
    will be loaded one at a time until all the examples have been yielded, at which point
    the next file will be loaded and used to yield examples, and so on. This means a large
    dataset can be broken into smaller chunks and this class can be used to load samples
    as if those files were one dataset while only utilizing the ram required for one chunk.
    
    Explanation about `batch_size` and `__len__()`:
    If the __len__ function is needed to be accurate then the `batch_size` must be specified
    when constructing objects of this class. PyTorch `DataLoader` objects will report accurate
    lengths by dividing the number of examples in the dataset by the batch size only if the
    dataset if not an `IterableDataset`. If the dataset is an `IterableDataset` then a `DataLoader`
    will simply ask the dataset for its length, without diving by the batch size, because
    in some cases the length of an `IterableDataset` might be difficult or impossible to determine.
    However, in this case the number of examples (length of dataset) is known. The division by
    batch size must happen in the dataset (for datasets of type `IterableDataset`) since the
    `DataLoader` will not calculate this.
    """

    # TODO: Add shuffling
    def __init__(self, files_list, shuffle=True, batch_size=1, verbose=False):
        super(FSIterableDataset).__init__()
        if shuffle:
            random.shuffle(files_list)  # happens in-place
        self.files_list = files_list
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.verbose = verbose
        self.total_length = None
        self.num_batches = None

    def __iter__(self):
        for data_file in self.files_list:
            if self.verbose:
                logger.info("Loading examples from " + str(data_file))
            dataset_section = torch.load(data_file)
            for example in dataset_section:
                yield example
                # input(example)
            # Clear memory usage before loading next file
            dataset_section = None
            gc.collect()
            del dataset_section
            gc.collect()

    def __len__(self):
        if self.num_batches:
            return self.num_batches
        else:
            logger.debug(
                "Calculating length of `IterableDataset` by loading each file, getting the length, and unloading, which is slow."
            )
            total_length = 0
            for data_file in self.files_list:
                dataset_section = torch.load(data_file)
                total_length += len(dataset_section)
            self.total_length = total_length

            # Calculate number of batches because the DataLoader `__len__` function directly
            # calls the `__len__` function of the dataset if the dataset is of type `IterableDataset`
            # DataLoader code: https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
            remainder_batch = 0 if (total_length % self.batch_size == 0) else 1
            num_batches = int(total_length / self.batch_size) + remainder_batch
            self.num_batches = num_batches
            return num_batches


class InputExample(object):
    def __init__(self, text, labels, guid=None, target=None):
        """A single training/test example for simple sequence classification.
        
        Arguments:
            text {list} -- The untokenized (for the appropriate model) text for the example.
                             Should be broken into sentences and tokens.
            labels {list} -- The labels of the example.
        
        Keyword Arguments:
            guid {int} -- A unique identification code for this example, not used. (default: {None})
            target {str} -- The ground truth abstractive summary. (default: {None})
        """
        self.guid = guid
        self.text = text
        self.labels = labels
        self.target = target

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in `[0, 1]`:
            Usually  `1` for tokens that are NOT MASKED, `0` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        labels: Labels corresponding to the input.
        sent_rep_token_ids: The locations of the sentence representation tokens.
        sent_lengths: A list of the lengths of each sentence in the `source` and `input_ids`.
        source: The actual source document as a list of sentences.
        target: The ground truth abstractive summary.
    """

    def __init__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        sent_rep_token_ids=None,
        sent_lengths=None,
        source=None,
        target=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels
        self.sent_rep_token_ids = sent_rep_token_ids
        self.sent_lengths = sent_lengths
        self.source = source
        self.target = target

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        # output = copy.deepcopy(self.__dict__)
        _dict = self.__dict__
        # removes empty and NoneType properties from `self.__dict__`
        output = {}
        for key, value in _dict.items():
            if value:
                output[key] = value
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SentencesProcessor:
    def __init__(self, name=None, labels=None, examples=None, verbose=False):
        """Create a `SentencesProcessor`
        
        Keyword Arguments:
            **All Optional
            name {str} -- a label for the `SentencesProcessor` object, used internally for saving if
                          a save name is not specified in `get_features()(default: {None})
            labels {list} -- the label that goes with each sample, can be a list of lists where
                             the inside lists are the labels for each sentence in the coresponding
                             example (default: {None})
            examples {list} -- list of `InputExample`s (default: {None})
            verbose {bool} -- log extra information (such as examples of processed data points) (default: {False})
        """
        self.name = name
        self.labels = [] if labels is None else labels
        self.examples = [] if examples is None else examples
        self.verbose = verbose

    def __len__(self):
        return len(self.examples)

    @classmethod
    def create_from_examples(cls, texts, labels=None, **kwargs):
        """
        Create a SentencesProcessor with **kwargs and add `texts` and labels` through add_examples()
        """
        processor = cls(**kwargs)
        processor.add_examples(texts, labels=labels)
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
        targets=None,
        overwrite_labels=False,
        overwrite_examples=False,
    ):
        """Primary method of adding example sets of texts, labels, ids, and targets to the SentencesProcessor
        
        Arguments:
            texts {list} -- A list of documents where each document is a list of sentences where each
                            sentence is a list of tokens. This is the output of `convert_to_extractive.py`
                            and is in the 'src' field for each doc. See `ExtractiveSummarizer.prepare_data`
                            (`main.py`).
        
        Keyword Arguments:
            labels {list} -- A list of the labels for each document where each label is a list of labels
                             where the index of the label coresponds with the index of the sentence in the
                             respective entry in `texts.` Similarly to `texts`, this is handled automatically
                             by `ExtractiveSummarizer.prepare_data`. (default: {None})
            ids {list} -- A list of ids for each document. Not used by `ExtractiveSummarizer`. (default: {None})
            oracle_ids {list} -- Similar to labels but is a list of indexes of the chosen sentences
                                 instead of a one-hot encoded vector. These will be converted to labels. (default: {None})
            targets {list} -- A list of the abstractive target for each document. (default: {None})
            overwrite_labels {bool} -- Replace any labels currently stored by the `SentencesProcessor`. (default: {False})
            overwrite_examples {bool} -- Replace any examples currently stored by the `SentencesProcessor`. (default: {False})
        
        Returns:
            list -- The examples as `InputExample`s that have been added
        """
        assert texts  # not an empty array
        assert labels is None or len(texts) == len(labels)
        assert ids is None or len(texts) == len(ids)
        assert not (labels and oracle_ids)
        assert isinstance(texts, list)

        if ids is None:
            ids = [None] * len(texts)
        if labels is None:
            if oracle_ids:  # convert `oracle_ids` to `labels`
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
        for idx, (text_set, label_set, guid) in enumerate(zip(texts, labels, ids)):
            if not text_set or not label_set:
                continue  # input()
            added_labels.append(label_set)
            if targets:
                example = InputExample(
                    guid=guid, text=text_set, labels=label_set, target=targets[idx]
                )
            else:
                example = InputExample(guid=guid, text=text_set, labels=label_set)
            examples.append(example)

        # Update examples
        if overwrite_examples:
            self.examples = examples
        else:
            self.examples.extend(examples)

        # Update labels
        if overwrite_labels:
            self.labels = added_labels
        else:
            self.labels += added_labels

        return self.examples

    def get_features_process(
        self,
        input_information,
        num_examples=0,
        tokenizer=None,
        bert_compatible_cls=True,
        sep_token=None,
        cls_token=None,
        create_sent_rep_token_ids=True,
        sent_rep_token_id=None,
        create_sent_lengths=True,
        create_segment_ids="binary",
        segment_token_id=None,
        create_source=False,
        max_length=None,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
        create_attention_mask=True,
        pad_ids_and_attention=True,
    ):
        """
        The process that actually creates the features. self.get_features() is the driving function,
        look there for a description of how this function works. This function only exists so that
        processing can easily be done in parallel using Pool.map.
        """
        ex_index, example, label = input_information
        if ex_index % 1000 == 0:
            logger.info(
                "Generating features for example "
                + str(ex_index)
                + "/"
                + str(num_examples)
            )
        if (
            bert_compatible_cls
        ):  # adds a '[CLS]' token between each sentence and outputs `input_ids`
            # convert `example.text` to array of sentences
            src_txt = [" ".join(sent) for sent in example.text]
            # separate each sentence with ' [SEP] [CLS] ' (or model equivalent tokens) and convert to string
            separation_string = " " + sep_token + " " + cls_token + " "
            text = separation_string.join(src_txt)
            # tokenize
            src_subtokens = tokenizer.tokenize(text)
            # select first `(max_length-2)` tokens (so the following line of tokens can be added)
            src_subtokens = src_subtokens[: (max_length - 2)]
            # add '[CLS]' to beginning and '[SEP]' to end (or model equivalent tokens)
            src_subtokens = [cls_token] + src_subtokens + [sep_token]
            # create `input_ids`
            input_ids = tokenizer.convert_tokens_to_ids(src_subtokens)
        else:
            input_ids = tokenizer.encode(
                example.text,
                add_special_tokens=True,
                max_length=min(max_length, tokenizer.max_len),
            )

        # Segment (Token Type) IDs
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

        # Sentence Representation Token IDs and Sentence Lengths
        sent_rep_ids = None
        sent_lengths = None
        if create_sent_rep_token_ids:
            # create list of indexes for the `sent_rep` tokens
            sent_rep_ids = [
                i for i, t in enumerate(input_ids) if t == sent_rep_token_id
            ]
            # truncate `label` to the length of the `cls_ids` aka the number of sentences
            label = label[: len(sent_rep_ids)]

            if create_sent_lengths:
                sent_lengths = [
                    sent_rep_ids[i] - sent_rep_ids[i - 1]
                    for i in range(1, len(sent_rep_ids))
                ]

        # Attention
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        if create_attention_mask:
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Padding
        # Zero-pad up to the sequence length.
        if pad_ids_and_attention:
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

            assert (
                len(input_ids) == max_length
            ), "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert (
                len(attention_mask) == max_length
            ), "Error with input length {} vs {}".format(
                len(attention_mask), max_length
            )

        if ex_index < 5 and self.verbose:
            logger.info("*** Example ***")
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
            if sent_lengths is not None:
                logger.info(
                    "sent_lengths: %s" % " ".join([str(x) for x in sent_lengths])
                )
            if create_attention_mask:
                logger.info(
                    "attention_mask: %s" % " ".join([str(x) for x in attention_mask])
                )
            logger.info("labels: %s (id = %s)" % (example.labels, label))

        # Return features
        # if the attention mask was created then add the mask to the returned features
        outputs = {
            "input_ids": input_ids,
            "labels": label,
            "token_type_ids": segment_ids,
            "sent_rep_token_ids": sent_rep_ids,
            "sent_lengths": sent_lengths,
            "target": example.target,
        }
        if create_attention_mask:
            outputs["attention_mask"] = attention_mask
        if create_source:
            # convert form individual tokens to only individual sentences
            source = [" ".join(sentence) for sentence in example.text]
            outputs["source"] = source

        return InputFeatures(**outputs)

    def get_features(
        self,
        tokenizer,
        bert_compatible_cls=True,
        create_sent_rep_token_ids=True,
        sent_rep_token_id=None,
        create_sent_lengths=True,
        create_segment_ids="binary",
        segment_token_id=None,
        create_source=False,
        n_process=2,
        max_length=None,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
        create_attention_mask=True,
        pad_ids_and_attention=True,
        return_type=None,
        save_to_path=None,
        save_to_name=None,
    ):
        """Convert the examples stored by the `SentencesProcessor` to features that can be used by
        a model. The following processes can be performed: tokenization, token type ids (to separate
        sentences),sentence representation token ids (the locations of each sentence representation
        token), sentence lengths, and the attention mask. Padding can be applied to the tokenized
        examples and the attention masks but it is recommended to instead use the `pad_batch_collate()`
        function so each batch is padded individually for efficiency (less zeros passed through model).
        
        Arguments:
            tokenizer {transformers.PreTrainedTokenizer} -- The tokenizer used to tokenize the examples.
        
        Keyword Arguments:
            bert_compatible_cls {bool} -- Adds '[CLS]' tokens in front of each sentence. This is useful
                                          so that the '[CLS]' token can be used to obtain sentence
                                          embeddings. This only works if the chosen model has the '[CLS]'
                                          token in its vocabulary. (default: {True})
            create_sent_rep_token_ids {bool} -- Option to create sentence representation token ids. This
                                                will store a list of the indexes of all the `sent_rep_token_id`s
                                                in the tokenized example. (default: {True})
            sent_rep_token_id {[type]} -- The token id that should be captured for each sentence (should have
                                          one per sentence and each should represent that sentence)
                                          (default: {'[CLS]' token if bert_compatible_cls else '[SEP]' token})
            create_sent_lengths {bool} -- Option to create a list of sentence lengths where each index in
                                          the list coresponds to the respective sentence in the example. (default: {True})
            create_segment_ids {str} -- Option to create segment ids (aka token type ids)
                                        See https://huggingface.co/transformers/glossary.html#token-type-ids for more info
                                        Set to either "binary", "sequential", or False.
                                        - `binary` alternates between 0 and 1 for each sentence.
                                        - `sequential` starts at 0 and increments by 1 for each sentence.
                                        - `False` does not create any segment ids.
                                        Note: Many pretrained models that accept token type ids use them
                                        for question answering ans related tasks where the model receives
                                        two inputs. Therefore, most models have a token type id vocabulary
                                        size of 2,which means they only have learned 2 token type ids. The
                                        "binary" mode exists so that these pretrained models can easily
                                        be used.
                                        (default: {"binary"})
            segment_token_id {str} -- The token id to be used when creating segment ids. Can be set to 'period'
                                      to treat periods as sentence separation tokens, but this is a terrible
                                      idea for obvious reasons. (default: {'[SEP]' token id})
            create_source {bool} -- Option to save the source text (non-tokenized) as a string. (default: {False})
            n_process {int} -- How many processes to use for multithreading for running get_features_process().
                               Set higher to run faster and set lower is you experience OOM issues. (default: {2})
            max_length {int} -- If `pad_ids_and_attention` is True then pad to this amount. (default: {tokenizer.max_len})
            pad_on_left {bool} -- Optionally, pad on the left instead of right. (default: {False})
            pad_token {int} -- Which token to use for padding the `input_ids`. (default: {0})
            mask_padding_with_zero {bool} -- Use zeros to pad the attention. Uses ones otherwise. (default: {True})
            create_attention_mask {bool} -- Option to create the attention mask. It is recommended to use
                                            the `pad_batch_collate()` function, which will automatically create
                                            attention masks and pad them on a per batch level. (default: {False if return_type == "lists" else True})
            pad_ids_and_attention {bool} -- Pad the `input_ids` with `pad_token` and attention masks
                                            with 0s or 1s deneding on `mask_padding_with_zero`. Pad both to
                                            `max_length`. (default: {False if return_type == "lists" else True})
            return_type {str} -- Either "tensors", "lists", or None. See "Returns" section below. (default: {None})
            save_to_path {str} -- The folder/directory to save the data to OR None to not save.
                                  Will save the type specified by `return_type` to disk. (default: {None})
            save_to_name {str} -- The name of the file to save. The extension '.pt' is automatically
                                  appended. (default: {'dataset_' + self.name + '.pt})
        
        Returns:
            If `return_type is None`: [list] -- The list of calculated features
            If `return_type == "tensors"`: [torch.TensorDataset] -- The features converted to tensors and stacked
                                                             such that features are grouped together into
                                                             individual tensors.
            If `return_type == "lists"`: [list] -- The recommended option. Exports each InputFeatures
                                                   object in the exported `features` list as a dictionary
                                                   and appends each dictionary to a list. Returns that list.
        """
        assert return_type in ["tensors", "lists"] or return_type is None
        if return_type == "lists":
            create_attention_mask = False
            pad_ids_and_attention = False
        else:  # if `return_type` is None  or "tensors"
            create_attention_mask = True
            pad_ids_and_attention = True

        if max_length is None:
            max_length = tokenizer.max_len

        # batch_length = max(len(input_ids) for input_ids in all_input_ids)

        if create_sent_rep_token_ids:
            if sent_rep_token_id == "sep":  # get the sep token id
                sent_rep_token_id = tokenizer.sep_token_id
            elif sent_rep_token_id == "cls":  # get the cls token id
                sent_rep_token_id = tokenizer.cls_token_id
            elif not sent_rep_token_id:  # if the `sent_rep_token_id` is not set
                # if using `bert_compatible_cls` then default to the `cls_token_id`
                if bert_compatible_cls:
                    sent_rep_token_id = tokenizer.cls_token_id
                else:  # otherwise, get the `sep_token_id`
                    sent_rep_token_id = tokenizer.sep_token_id

        if create_segment_ids:
            if segment_token_id == "period":  # get the token id for a "."
                segment_token_id = tokenizer.convert_tokens_to_ids(["."])[0]
            elif (
                not segment_token_id
            ):  # default to trying to get the `sep_token_id` if the `segment_token_id` is not set
                segment_token_id = tokenizer.sep_token_id

        features = []
        pool = Pool(n_process)
        _get_features_process = partial(
            self.get_features_process,
            num_examples=len(self.labels),
            tokenizer=tokenizer,
            bert_compatible_cls=bert_compatible_cls,
            sep_token=tokenizer.sep_token,
            cls_token=tokenizer.cls_token,
            create_sent_rep_token_ids=create_sent_rep_token_ids,
            sent_rep_token_id=sent_rep_token_id,
            create_sent_lengths=create_sent_lengths,
            create_segment_ids=create_segment_ids,
            segment_token_id=segment_token_id,
            create_source=create_source,
            max_length=max_length,
            pad_on_left=pad_on_left,
            pad_token=pad_token,
            mask_padding_with_zero=mask_padding_with_zero,
            create_attention_mask=create_attention_mask,
            pad_ids_and_attention=pad_ids_and_attention,
        )

        for rtn_features in pool.map(
            _get_features_process,
            zip(range(len(self.labels)), self.examples, self.labels),
        ):
            features.append(rtn_features)

        pool.close()
        pool.join()

        if not return_type:
            return features
        elif return_type == "tensors":
            final_tensors = []

            all_input_ids = torch.tensor(
                [f.input_ids for f in features], dtype=torch.long
            )
            final_tensors.append(all_input_ids)
            all_attention_masks = torch.tensor(
                [f.attention_mask for f in features], dtype=torch.long
            )
            final_tensors.append(all_attention_masks)
            all_labels = torch.tensor(
                pad([f.labels for f in features], 0), dtype=torch.long
            )
            final_tensors.append(all_labels)

            if create_segment_ids:
                all_token_type_ids = torch.tensor(
                    pad([f.token_type_ids for f in features], 0), dtype=torch.long
                )
                final_tensors.append(all_token_type_ids)
            # Pad sentence representation token ids (`sent_rep_token_ids`)
            if create_sent_rep_token_ids:
                all_sent_rep_token_ids = torch.tensor(
                    pad([f.sent_rep_token_ids for f in features], -1), dtype=torch.long
                )
                all_sent_rep_token_ids_masks = ~(all_sent_rep_token_ids == -1)
                all_sent_rep_token_ids[all_sent_rep_token_ids == -1] = 0
                final_tensors.append(all_sent_rep_token_ids)
                final_tensors.append(all_sent_rep_token_ids_masks)

                if create_sent_lengths:
                    all_sent_lengths = torch.tensor(
                        pad([f.sent_lengths for f in features], 0), dtype=torch.long
                    )
                    final_tensors.append(all_sent_lengths)

            dataset = TensorDataset(*final_tensors)

        elif return_type == "lists":
            dataset = [example.to_dict() for example in features]

            # dataset = {}
            # dataset["all_input_ids"] = [f.input_ids for f in features]
            # dataset["all_attention_masks"] = [f.attention_mask for f in features]
            # dataset["all_labels"] = [f.labels for f in features]
            # if create_segment_ids:
            #     dataset["all_token_type_ids"] = [f.token_type_ids for f in features]
            # if create_sent_rep_token_ids:
            #     dataset["all_sent_rep_token_ids"] = [
            #         f.sent_rep_token_ids for f in features
            #     ]

        if save_to_path:
            final_save_name = save_to_name if save_to_name else ("dataset_" + self.name)
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