import flax
import jax.numpy as jnp

from typing import List, Union, Dict
from torch.utils.data.dataloader import (DataLoader, RandomSampler,
                                         SequentialSampler)
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset


@flax.struct.dataclass
class FlaxDataCollator:
    """
    Args:
        tokenizer: The tokenizer for encode sample
        source_len (:obj:int): The expected source_len after masking
        target_len (:obj:int): The expected target_len after masking
        batch_size (:obj:int): How many samples per batch to load
        text_column (:obj:int): Name of input column
        target_column (obj:str): Name of labels column
    """
    tokenizer: PreTrainedTokenizerBase
    source_len: int
    target_len: int
    batch_size: int
    text_column: str
    target_column: str
    do_train: bool
    do_eval: bool
    do_test: bool
    train_file: str
    eval_file: str
    test_file: str

    def __call__(self):
        dataloader = {}
        if self.do_train and self.train_file is not None:
            train_dataset = self.load_data("train", self.train_file)
            dataloader['train'] = self.get_dataloader(train_dataset,
                                                      shuffle_flag=True)

        if self.do_eval and self.eval_file is not None:
            eval_dataset = self.load_data("val", self.eval_file)
            dataloader['eval'] = self.get_dataloader(eval_dataset)

        if self.do_test and self.test_file is not None:
            test_dataset = self.load_data("test", self.test_file)
            dataloader['test'] = self.get_dataloader(test_dataset)

        return dataloader

    def load_data(self, key: str, data_file: Union[str, List[str]]):
        """
        :param data_path (:obj:str): The path to file data (txt, json)
        """

        data_files = {key: data_file}
        extension = data_file.split(".")[-1]
        dataset = load_dataset(extension, data_files=data_files, split=key)
        return dataset

    def data_collator(self, batch: Dict[str, str]) -> Dict[str, jnp.ndarray]:
        """
        Data collator used for language modeling.
        :param batch:  how many samples per batch to load
        :return:
        """
        inputs = [example[self.text_column] for example in batch]
        targets = [example[self.target_column] for example in batch]

        inp_tokens = self.tokenizer.batch_encode_plus(
            inputs,
            max_length= self.source_len if self.source_len > 0 else None,
            padding="longest",
            truncation= True if self.source_len > 0 else False,
            return_tensors="pt",)

        tgt_tokens = self.tokenizer.batch_encode_plus(
            targets,
            max_length= self.target_len if self.target_len > 0 else None,
            padding="longest",
            truncation=True if self.target_len > 0 else False,
            return_tensors="pt",)

        tgt_ids = tgt_tokens["input_ids"]
        tgt_mask = tgt_tokens["attention_mask"].bool()
        tgt_ids = tgt_ids.masked_fill(~tgt_mask, -100)

        return {
            "inputs_ids" : jnp.array(inp_tokens['input_ids']),
            "attention_mask" : jnp.array(inp_tokens['attention_mask']),
            "labels" : jnp.array(tgt_ids)
        }

    def get_dataloader(self, dataset,
                       shuffle_flag: bool = False) -> DataLoader:
        """

        :param dataset: (Dataset): dataset from which to load the data.
        :param shuffle_flag: set to ``True`` to have the data reshuffled
                at every epoch (default: ``False``).
        :return: a dataset
        """

        sampler = RandomSampler(dataset) if shuffle_flag else SequentialSampler(dataset)
        dataloader = DataLoader(dataset,
                                sampler= sampler,
                                collate_fn=self.data_collator,
                                batch_size=self.batch_size)

        return dataloader

if __name__ == '__main__':
    # test
    from transformers import AutoTokenizer

    tokenizers = AutoTokenizer.from_pretrained("prakharz/DIAL-BART0")

    data_collator = FlaxDataCollator(tokenizer=tokenizers,
                                     source_len=0,
                                     target_len=0,
                                     batch_size=4,
                                     text_column="prompt",
                                     target_column="output",
                                     do_train=True,
                                     do_eval=True,
                                     do_test=True,
                                     train_file="/content/data/train_converted.json",
                                     eval_file="/content/data/valid_converted.json",
                                     test_file="/content/data/test_converted.json")

    dataloader = data_collator.__call__()
    for step, batch in enumerate(dataloader['train']):
        print(batch)
        break