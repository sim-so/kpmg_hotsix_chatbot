from functools import partial
from datasets import load_dataset
from transformers import AutoTokenizer

class DataloaderForNER():
    def __init__(self, tokenizer_name, max_length, label_type=None):
        """
        label_type 
            - None or "all" : use 15 main tags.
            - TR : use only TR tag.
            - tr : use 6 tags in TR category.
            - tr_science : use "TR_SCIENCE" tag only.
            - list of tags : use tags in the list.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.short_label = True

        if (label_type is None) or (label_type == "all"):
            tag_list = ["PS", "FD", "TR", "AF", "OG", "LC", "CV", "DT", "TI", "QT", "EV", "AM", "PT", "MT", "TM"]                
        elif label_type == "TR":
            tag_list = ["TR"]
        elif label_type == "tr":
            tag_list = ["TR_SCIENCE", "TR_SOCIAL_SCIENCE", "TR_MEDICINE", "TR_ART", "TR_HUMANITIES", "TR_OTHERS"]
            self.short_label = False
        elif label_type == "tr_science":
            tag_list = ["TR_SCIENCE"]
            self.short_label = False
        elif isinstance(label_type, list):
            tag_list = label_type
            self.short_label = False
        else:
            raise ValueError("Not supported type.")
        bio_tag_list = []
        for tag in tag_list:
            bio_tag_list.append("B-"+tag)
            bio_tag_list.append("I-"+tag)
        self.label_to_id = {bio_tag: idx for idx, bio_tag in enumerate(bio_tag_list, 1)}
        self.label_to_id['O'] = 0

    def load_data(self):
        dataset = load_dataset("csv", data_files=dict(train="data/train_data.csv", eval="data/eval_data.csv"), sep="\t")
        train_dataset = dataset["train"].map(partial(self.BIO_tagging), batched=False).remove_columns(["data", "label"])
        eval_dataset = dataset["eval"].map(partial(self.BIO_tagging), batched=False).remove_columns(["data", "label"])
        return train_dataset, eval_dataset

    def BIO_tagging(self, example):
        encoded = self.tokenizer(
            example["data"], 
            return_token_type_ids=False,
            return_offsets_mapping=True,
            max_length=self.max_length,
            truncation=True)
        label_list = eval(example["label"])
        label_offset_list = [(label["begin"], label["end"]) for label in label_list]
        label_tag_list = [label["label"][:2] if self.short_label else label["label"] for label in label_list]
        label_idx = 0
        is_begin = True
        label_sequence = []
        for token_idx, offset_map in enumerate(encoded["offset_mapping"]):
            token_begin, token_end = offset_map
            if token_end - token_begin > 1 and example["data"][token_begin] == " ":
                token_begin += 1
            if token_end == 0:
                label_sequence.append('O')
                continue
            if label_idx < len(label_offset_list):
                label_start, label_end = label_offset_list[label_idx]
                if label_end < token_begin:
                    is_begin = True
                    label_idx = label_idx + 1 if label_idx + 1 < len(label_offset_list) else label_idx
                    label_start, label_end = label_offset_list[label_idx]
                if label_start <= token_begin and label_end > token_begin:
                    label_tag = label_tag_list[label_idx]
                    if is_begin:
                        label_sequence.append('B-'+label_tag)
                        is_begin = False
                    else:
                        label_sequence.append('I-'+label_tag)
                else:
                    is_begin = True
                    label_sequence.append('O')
            else:
                label_sequence.append('O')

        label_sequence_id = [self.label_to_id.get(label, 0) for label in label_sequence]
        encoded["labels"] = label_sequence_id
        return encoded