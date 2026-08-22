import numpy as np
from typing import List, Dict

from model import Model
from .wsd_model import WSD_Model

#-- PyTorch
import torch

#-- NLTK and WordNet
import nltk
from nltk.corpus import wordnet as wn
print("-- NLTK Imported --")
nltk.download('wordnet')
print(" - Wordnet Version: {}".format(wn.get_version()))



def build_model(device: str) -> Model:
    # return RandomBaseline()
    return TestModel(checkpoint_path=f"model/epoch_weak.ckpt",
                     coarse_fine_path=f"model/coarse-fine_map.data", weak_supervision=True)


class RandomBaseline(Model):
    def __init__(self):
        # Load your models/tokenizer/etc. that only needs to be loaded once when doing inference
        pass
    def predict(self, sentences: List[Dict]) -> List[List[str]]:
        return [[np.random.choice(candidates) for candidates in sentence_data["candidates"].values()]
                for sentence_data in sentences]


class TestModel(Model):
    def __init__(self, checkpoint_path:str, coarse_fine_path:str, weak_supervision:bool=True): 
        # Load your models/tokenizer/etc. that only needs to be loaded once when doing inference
        
        # DEVICE INSTANTIATION
        if torch.cuda.is_available() == True: self.device = torch.device("cuda")
        else: self.device = torch.device("cpu")      
        print(f"Device used: {self.device}")

        # COARSE-FINE MAPPING for WSD
        self.coarse_fine_mapping = torch.load(coarse_fine_path)["coarse-fine"]
        self.weak_supervision = weak_supervision
        
        # MODEL INSTANTIATION
        print("-- Loading Model from Checkpoint --")
        print(f" - Checkpoint Path: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        print(" - Model Instantiation")
        self.wsd_model = WSD_Model(fine_tuning=False,
                                   token_path="model/bert_tokenizer", model_path="model/bert_model") 
        self.wsd_model.load_state_dict(ckpt["state_dict"])
        self.wsd_model.to(device=self.device)      
        print("  - Last Training Checkpoint Loaded")
        
        self.wsd_model.eval()
        print("  - Model in EVALUATION Mode")    
        print("-- Model Loaded Correctly --")

    def predict(self, sentences: List[Dict]) -> List[List[str]]:
        model_predictions = []
        for sentence in sentences:
            context_sentence = sentence["words"]
            target_words = list(sentence["instance_ids"].keys())
            
            sentence_predictions = []
            for t_w in target_words:
                possible_homonomies = sentence["candidates"][t_w]       # List

                context_gloss_pairs = []
                for hom_i in possible_homonomies:
                    try: fine_candidates = [key for key in self.coarse_fine_mapping[hom_i][0].keys()]
                    except:
                        fine_candidates = wn.synsets(hom_i)
                        fine_candidates = [elem.name() for elem in fine_candidates]
                    new_sentence = context_sentence.copy()
                    for elem in fine_candidates:
                        syn = wn.synset(elem)
                        gloss = syn.definition()
                        if self.weak_supervision == True:
                            old_word = context_sentence[int(t_w)]
                            new_old_word = '"' + old_word + '"'         # Weak-Supervision on target word
                            new_sentence[int(t_w)] = new_old_word
                            gloss_sentence = "[SEP] " + old_word + " : " + gloss
                            new_sentence.append(gloss_sentence)
                            context_gloss_pairs.append((new_sentence))
                        else:
                            gloss_sentence = "[SEP] " + gloss
                            new_sentence.append(gloss_sentence)
                            context_gloss_pairs.append((new_sentence))
                encoded_sentences = self.wsd_model.tokenizer([sent for sent in context_gloss_pairs],
                                                            padding="max_length", return_tensors="pt", truncation=True,
                                                            max_length=self.wsd_model.max_seq_length, is_split_into_words=True).to(device=self.device)              
                with torch.no_grad():
                    prediction = self.wsd_model(encoded_sentences)
                argmax_idx = torch.argmax(prediction, dim=0).item()
                sentence_predictions.append(possible_homonomies[argmax_idx])
            model_predictions.append(sentence_predictions)

        return model_predictions
    