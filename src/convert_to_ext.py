import os
import re
import json
import gzip
import glob
import logging
import spacy
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from time import time

logger = logging.getLogger(__name__)

# 步骤
# 运行 cnn/dm 处理脚本以获取训练、测试、验证二进制文本文件
# 对于每个 bin：
#   加载所有数据，每行是列表中的一个条目
#   对于每个文档（行）（并行化）：
#       对源文件和目标文件中的行进行分词
#       通过 oracle_id 算法处理源和目标
#       在数据处理器中运行当前的 preprocess_examples() 函数（数据清理）
#       返回源（作为句子列表）和目标
#   在 map() 循环中：将每个（源、目标、标签）附加到变量，并在完成后保存（作为 cnn_dm_extractive）

# BertSum：
# 1. 将所有文件标记化为标记化的 json 版本
# 2. 将 json 拆分为源和目标，并将故事连接成 `shard_size` 数量的块
# 3. 处理以获得每个块的提取摘要和标签
# 4. 将每个处理过的块保存为包含处理值的字典列表


def read_in_chunks(file_object, chunk_size=5000):
    """ Read a file line by line but yield chunks of `chunk_size` number of lines at a time. """
    # https://stackoverflow.com/a/519653
    # 从 1 开始计数，因为零对任何数的模都是零
    current_line_num = 1
    lines = []
    for line in file_object:
        # use `(chunk_size + 1)` because each iteration starts at 1
        if current_line_num % (chunk_size + 1) == 0:
            yield lines
            # reset the `lines` so a new chunk can be yielded
            lines.clear()
            # Essentially adds one twice so that each interation starts counting at one.
            # This means each yielded chunk will be the same size instead of the first
            # one being 5000 then every one after being 5001, for example.
            current_line_num += 1
        lines.append(line)
        current_line_num += 1


def convert_to_extractive_driver(args):
    """ 驱动程序函数，将抽象摘要数据集转换为提取式数据集。
    抽象数据集必须为每个拆分格式化为两个文件：源文件和目标文件。
    示例文件列表：["train.source", "train.target", "val.source", "val.target"]
    """
    # 默认情况下，如果未指定输出目录，则输出到输入数据目录
    if not args.base_output_path:
        args.base_output_path = args.base_path

    # load spacy english small model with the "tagger" and "ner" disabled since
    # we only need the "tokenizer" and "parser"
    # more info: https://spacy.io/usage/processing-pipelines
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "ner"])

    # 对于每个拆分
    for name in tqdm(
        args.split_names, total=len(args.split_names), desc="Dataset Split"
    ):
        # 获取源和目标路径
        source_file_path = os.path.join(args.base_path, (name + "." + args.source_ext))
        target_file_path = os.path.join(args.base_path, (name + "." + args.target_ext))
        logger.info("Opening source and target " + str(name) + " files")

        with open(source_file_path, "r") as source_file, open(
            target_file_path, "r"
        ) as target_file:
            if args.shard_interval:  # if sharding is enabled
                # 获取要处理的示例数量
                target_file_len = sum(1 for line in target_file)
                # 在获取长度后将指针重置回开头
                target_file.seek(0)

                # 找到循环将运行多长时间
                tot_num_interations = int(target_file_len / args.shard_interval)

                # 默认情况下，假设没有先前的分片（即不恢复）
                last_shard = 0
                if args.resume:
                    num_lines_read, last_shard = resume(
                        args.base_output_path, name, args.shard_interval
                    )

                    # 如果已读取行并且已将分片写入磁盘
                    if num_lines_read:
                        logger.info("Resuming to line " + str(num_lines_read))
                        # 将源和目标都寻址到下一行
                        seek_files([source_file, target_file], num_lines_read)

                        # 减去已经创建的分片数量
                        tot_num_interations -= int(last_shard)
                    else:  # no shards on disk
                        logger.warn("Tried to resume but no shards found on disk")

                for piece_idx, (source_docs, target_docs) in tqdm(
                    enumerate(
                        zip(
                            read_in_chunks(source_file, args.shard_interval),
                            read_in_chunks(target_file, args.shard_interval),
                        )
                    ),
                    total=tot_num_interations,
                    desc="Shards",
                ):
                    piece_idx += last_shard  # effective if resuming (offsets the index)
                    convert_to_extractive_process(
                        args, nlp, source_docs, target_docs, name, piece_idx
                    )
            else:
                source_docs = [line.strip() for line in source_file]
                target_docs = [line.strip() for line in target_file]
                convert_to_extractive_process(args, nlp, source_docs, target_docs, name)


def convert_to_extractive_process(
    args, nlp, source_docs, target_docs, name, piece_idx=None
):
    """ 主过程，将抽象摘要数据集转换为提取式。
    对源和目标文档进行分词，获取 `oracle_ids`，拆分为 `source` 和 `labels`，并保存处理后的数据。
    """
    # 对源和目标文档进行分词
    # 每个步骤在 `args.n_process` 线程上并行运行，批大小为 `args.batch_size`
    source_docs_tokenized = tokenize(
        nlp,
        source_docs,
        args.n_process,
        args.batch_size,
        name=(" " + name + "-" + args.source_ext),
    )
    target_docs_tokenized = tokenize(
        nlp,
        target_docs,
        args.n_process,
        args.batch_size,
        name=(" " + name + "-" + args.target_ext),
    )

    # 设置常量 `oracle_mode`
    _example_processor = partial(example_processor, oracle_mode=args.oracle_mode)

    dataset = []
    pool = Pool(args.n_process)
    logger.info("Processing " + name)
    t0 = time()
    for idx, preprocessed_data in enumerate(
        pool.map(_example_processor, zip(source_docs_tokenized, target_docs_tokenized),)
    ):
        if preprocessed_data is not None:
            # preprocessed_data is (source_doc, labels)
            dataset.append(
                {"src": preprocessed_data[0], "labels": preprocessed_data[1]}
            )

    pool.close()
    pool.join()
    logger.info("Done in " + str(time() - t0) + " seconds")

    if args.shard_interval:
        split_output_path = os.path.join(
            args.base_output_path, (name + "." + str(piece_idx) + ".json")
        )
    else:
        split_output_path = os.path.join(args.base_output_path, (name + ".json"))
    save(dataset, split_output_path, compression=args.compression)


def resume(output_path, split, chunk_size):
    """ 查找最后创建的分片并返回读取的总行数和最后的分片编号。 """
    glob_str = os.path.join(output_path, (split + ".*.json"))
    all_json_in_split = glob.glob(glob_str)

    if not all_json_in_split:  # if no files found
        return None

    # get the second match because the first one includes the "." and the text between
    # but we only want the text between. also convert to int so max() operator works
    shard_file_idxs = [
        int(re.search("\.((.*))\.", a).group(2)) for a in all_json_in_split
    ]

    last_shard = int(max(shard_file_idxs)) + 1  # because the file indexes start at 0

    num_lines_read = chunk_size * last_shard
    # `num_lines_read` is the number of lines read if line indexing started at 1
    # therefore, this number is the number of the next line wanted
    return num_lines_read, last_shard


def seek_files(files, line_num):
    """ 将一组文件寻址到行号 `line_num` 并返回文件。 """
    rtn_file_objects = []
    for file_object in files:
        line_offset = []
        offset = 0
        for idx, line in enumerate(file_object):
            if idx >= line_num:
                break
            offset += len(line)
        file_object.seek(0)

        file_object.seek(offset)
        rtn_file_objects.append(file_object)
    return rtn_file_objects


def save(json_to_save, output_path, compression=False):
    """ 保存 `json_to_save` 到 `output_path`，并根据 `compression` 指定可选的 gzip 压缩 """
    logger.info("Saving to " + str(output_path))
    if compression:
        # https://stackoverflow.com/a/39451012
        json_str = json.dumps(json_to_save)
        json_bytes = json_str.encode("utf-8")
        with gzip.open((output_path + ".gz"), "w") as save:
            save.write(json_bytes)
    else:
        with open(output_path, "w") as save:
            save.write(json.dumps(json_to_save))


def tokenize(nlp, docs, n_process=5, batch_size=100, name=""):
    """ 使用 spacy 进行分词并拆分为句子和标记 """
    tokenized = []

    for idx, doc in tqdm(
        enumerate(nlp.pipe(docs, n_process=n_process, batch_size=batch_size,)),
        total=len(docs),
        desc="Tokenizing" + name,
    ):
        tokenized.append(doc)

    logger.info("Splitting into sentences and tokens and converting to lists")
    t0 = time()

    doc_sents = [list(doc.sents) for doc in tokenized]
    del tokenized
    sents = [
        list(list(token.text for token in sentence) for sentence in doc)
        for doc in doc_sents
    ]
    del doc_sents

    logger.info("Done in " + str(time() - t0) + " seconds")
    # `sents` 是一个文档数组，其中每个文档是一个句子数组，每个句子是一个标记数组
    return sents


def example_processor(args, oracle_mode="greedy"):
    """ 创建 `oracle_ids`，将其转换为 `labels` 并运行 preprocess()。 """
    source_doc, target_doc = args
    if oracle_mode == "greedy":
        oracle_ids = greedy_selection(source_doc, target_doc, 3)
    elif oracle_mode == "combination":
        oracle_ids = combination_selection(source_doc, target_doc, 3)

    # `oracle_ids` to labels
    labels = [0] * len(source_doc)
    for l in oracle_ids:
        labels[l] = 1

    preprocessed_data = preprocess(source_doc, labels)
    return preprocessed_data


def preprocess(
    example,
    labels,
    min_sentence_ntokens=5,
    max_sentence_ntokens=200,
    min_example_nsents=3,
    max_example_nsents=100,
):
    """ 删除过长或过短的句子，并删除句子数量过少或过多的示例。 """
    # pick the sentence indexes in `example` if they are larger then `min_sentence_ntokens`
    idxs = [i for i, s in enumerate(example) if (len(s) > min_sentence_ntokens)]
    # truncate selected source sentences to `max_sentence_ntokens`
    example = [example[i][:max_sentence_ntokens] for i in idxs]
    # only pick labels for sentences that matched the length requirement
    labels = [labels[i] for i in idxs]
    # truncate entire source to max number of sentences (`max_example_nsents`)
    example = example[:max_example_nsents]
    # perform above truncation to `labels`
    labels = labels[:max_example_nsents]

    # if the example does not meet the length requirement then return None
    if len(example) < min_example_nsents:
        return None
    return example, labels


# Section Methods (to convert abstractive summary to extractive)
# Copied from https://github.com/nlpyang/BertSum/blob/9aa6ab84faf3a50724ce7112c780a4651de289b0/src/prepro/data_builder.py
def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r"[^a-zA-Z0-9 ]", "", s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(" ".join(abstract)).split()
    sents = [_rouge_clean(" ".join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations(
            [i for i in range(len(sents)) if i not in impossible_sents], s + 1
        )
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)["f"]
            rouge_2 = cal_rouge(candidates_2, reference_2grams)["f"]

            rouge_score = rouge_1 + rouge_2
            if s == 0 and rouge_score == 0:
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r"[^a-zA-Z0-9 ]", "", s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(" ".join(abstract)).split()
    sents = [_rouge_clean(" ".join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if i in selected:
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)["f"]
            rouge_2 = cal_rouge(candidates_2, reference_2grams)["f"]
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if cur_id == -1:
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i : i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert an Abstractive Summarization Dataset to the Extractive Task"
    )

    parser.add_argument(
        "base_path", metavar="DIR", type=str, help="path to data directory"
    )
    parser.add_argument(
        "--base_output_path",
        type=str,
        default=None,
        help="path to output processed data (default is `base_path`)",
    )
    parser.add_argument(
        "--split_names",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        nargs="+",
        help="which splits of dataset to process",
    )
    parser.add_argument(
        "--source_ext", type=str, default="source", help="extension of source files"
    )
    parser.add_argument(
        "--target_ext", type=str, default="target", help="extension of target files"
    )
    parser.add_argument(
        "--oracle_mode",
        type=str,
        default="greedy",
        choices=["greedy", "combination"],
        help="method to convert abstractive summaries to extractive summaries",
    )
    parser.add_argument(
        "--shard_interval",
        type=int,
        default=None,
        help="how many examples to include in each shard of the dataset (default: no shards)",
    )
    parser.add_argument(
        "--n_process",
        type=int,
        default=6,
        help="number of processes for multithreading",
    )
    parser.add_argument(
        "--batch_size", type=int, default=50, help="number of batches for tokenization"
    )
    parser.add_argument(
        "--compression",
        action="store_true",
        help="use gzip compression when saving data",
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume from last shard",
    )
    parser.add_argument(
        "-l",
        "--log",
        dest="logLevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: 'Info').",
    )
    args = parser.parse_args()

    if args.resume and not args.shard_interval:
        parser.error(
            "Resuming requires both shard mode (--shard_interval) to be enabled and shards to be created. Must use same 'shard_interval' that was used previously to create the files to be resumed from."
        )

    # Setup logging config
    logging.basicConfig(
        format="%(asctime)s|%(name)s|%(levelname)s> %(message)s",
        level=logging.getLevelName(args.logLevel),
    )

    convert_to_extractive_driver(args)