from nltk.corpus.reader.wordnet import Synset

def look_for_synset_definition(syn:Synset):
    definition = syn.definition()
    return definition

def look_for_synset_key(syn:Synset):
    #print(f"  - WordNet Synset Lemmas Key: {syn.lemmas()[0].key()}")
    synset_key = syn.lemmas()[0].key()
    return synset_key