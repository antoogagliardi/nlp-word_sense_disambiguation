from typing import Tuple
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import BertTokenizerFast
from nltk.corpus import wordnet as wn

from src.wordnet import look_for_synset_definition, look_for_synset_key


class WSD_Dataset(Dataset):
    """A custom dataset class for Word Sense Disambiguation (WSD) tasks.
       This class extends the `torch.utils.data.Dataset` class and provides methods to load and preprocess data for WSD tasks.

    Attributes:
        coarse_data (dict): A dictionary cotaining coarse-grain data of our samples (training, validation or test).
        fine_data (dict): A dictionary cotaining fine-grain our samples (training, validation or test).
        coarse_fine_mapping (dict): A list of target labels corresponding to the input data.

    Methods:
        __init__(self, data, targets): Initializes the WSD dataset
        __len__(self): Returns the number of samples in the dataset.
        __getitem__(self, index): Retrieves a specific sample from the dataset. Importan: in this case that each sample is mapped
            with a key in the dictionary that is a string. So basically we need two dictionaries, key_id_mapping and id_key_mapping, 
            in order to match the gith element during the retrieving procedure of the sample_i
    """
    def __init__(self, coarse_data:dict, fine_data:dict, coarse_fine_mapping:dict, weak_supervision:bool=True):
        super(WSD_Dataset, self).__init__()

        # Data Dictionary
        self.coarse_data = coarse_data
        self.fine_data = fine_data
        # Fine-Coarse Grain Mapping Data
        self.coarse_fine_mapping = coarse_fine_mapping

        # Key Sentence Dictionary-ID Mapping
        self.key_id_mapping, self.id_key_mapping = self.keys_ids_mapping() # To retrieve and object from an index during the training, because the main key is a string

        # Dataset Construction
        self.dataset = self.construct_dataset(weak_superivison=weak_supervision)
        print("Length of Dataset: ", len(self.dataset))

    # -- Mapping Function
    def keys_ids_mapping(self) -> Tuple[dict, dict]:
        key_id_mapping = {}
        id_key_mapping = {}
        with tqdm(range(len(self.coarse_data.items())), desc="Key Mapping Creation") as pbar:
            for i, (key) in zip(pbar, self.coarse_data.keys()):
                id = len(key_id_mapping)
                key_id_mapping[key] = id
                id_key_mapping[id] = key

        return key_id_mapping, id_key_mapping

    def construct_dataset(self, weak_superivison:bool=True):
        dataset = []
        with tqdm(range(len(self.coarse_data.keys())), desc="Gloss-Context Pair Dataset Creation") as pbar:
            for idx, i in zip(pbar, range(len(self.coarse_data.keys()))):
                sent = self.id_key_mapping[i]

                context_sentence = self.coarse_data[sent]["words"]
                ambiguous_words = list(self.coarse_data[sent]["instance_ids"].keys())
                pbar.set_postfix(SENTENCE=sent,
                                TARGET_WORDS=len(ambiguous_words))
                pbar.update(0)
                context_gloss_pairs = []
                for t_w in ambiguous_words:
                    right_sense = self.coarse_data[sent]["senses"][t_w]                 # List
                    possible_homonomies = self.coarse_data[sent]["candidates"][t_w]     # List

                    for hom_i in possible_homonomies:
                        try: fine_candidates = [key for key in self.coarse_fine_mapping[hom_i][0].keys()]
                        except:
                            fine_candidates = wn.synsets(hom_i)
                            fine_candidates = [elem.name() for elem in fine_candidates]
                        for elem in fine_candidates:
                            new_sentence = context_sentence.copy()
                            syn = wn.synset(elem)
                            gloss = syn.definition()
                            syn_key = look_for_synset_key(syn)
                            pbar.set_postfix(SENTENCE=sent,
                                        TARGET_WORDS=len(ambiguous_words),
                                        SYNSET_KEY = syn_key)
                            pbar.update(0)
                            if weak_superivison == True:
                                old_word = context_sentence[int(t_w)]
                                new_old_word = '"' + old_word + '"'
                                new_sentence[int(t_w)] = new_old_word
                                gloss_sentence = "[SEP] " + old_word + " : " + gloss
                                new_sentence.append(gloss_sentence)
                            else:
                                gloss_sentence = "[SEP] " + look_for_synset_definition(syn)
                                new_sentence.append(gloss_sentence)

                            if hom_i == right_sense[0]: context_gloss_pairs.append((new_sentence, "YES", syn_key))
                            else: context_gloss_pairs.append((new_sentence, "NO", syn_key))
                for elem in context_gloss_pairs:
                    dataset.append(elem)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # ---- Reterieve the right item
        item = index

        return self.dataset[item]



class WSD_DataModule(pl.LightningDataModule):
    def __init__(self, train_dataset:Dataset, valid_dataset:Dataset, test_dataset:Dataset, batch_size:int=8, train_shuffle:bool=True):
        super(WSD_DataModule, self).__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.max_len = self.tokenizer.max_model_input_sizes["bert-base-uncased"]
        self.label_id_mappping = {"NO":0.0,"YES":1.0}
        self.id_label_mapping = {0.0:"NO", 1.0:"YES"}

        self.batch_size = batch_size
        self.training_shuffle = train_shuffle

        self.training_dataset = train_dataset
        self.validation_dataset = valid_dataset
        self.test_dataset = test_dataset

    def encode_collate_fn(self, samples):
        encoded_sentences = self.tokenizer([sent for sent, label, syn_key in samples],
                                        padding="max_length", return_tensors="pt", truncation=True,
                                        max_length=self.max_len, is_split_into_words=True)
        encoded_labels = torch.FloatTensor([self.label_id_mappping[label] for sent, label, syn_key in samples])
        return encoded_sentences, encoded_labels

    def train_dataloader(self):
        if self.training_dataset != None:
            print("-- Preparing DataLoader for the training set --")
            print(f" - Shuffe Training Dataset: {True if self.training_shuffle == True else False}")
            train_dataloader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True if self.training_shuffle == True else False,
                                          collate_fn=self.encode_collate_fn)
            print(" - Training DataLoader Ready")
            return train_dataloader
        else:
            print("No Training Data Available")

    def val_dataloader(self):
        if self.validation_dataset != None:
            print("-- Preparing DataLoader for the validation set --")
            valid_dataloader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False,
                                          collate_fn=self.encode_collate_fn)
            print(" - Validation DataLoader Ready")
            return valid_dataloader
        else:
            print("No Validation Data Available")


    def test_dataloader(self):
        if self.test_dataset != None:
            print("-- Preparing DataLoader for the test set --")
            test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                                         collate_fn=self.encode_collate_fn)
            print(" - Test DataLoader Ready")
            return test_dataloader
        else:
            print("No Testing Data Available")
