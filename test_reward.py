from typing import List, Callable, Union
import os
import random
import numpy as np
from multiprocessing import Pool
from time import time

import nltk
from nltk.translate.bleu_score import SmoothingFunction

from fast_bleu import BLEU
    
class SelfBleuReward(object):

    def __init__(self, 
                 grams: List[int] = [3, 4, 6],
                 tokenizer: Callable = nltk.word_tokenize) -> None: 
        self.grams = grams
        self.tokenizer = tokenizer
        self.time_last_run = time()
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')

        self.weights = {f"{n}-gram": ([1. / n] * n) for n in self.grams}
        # Initialize BLEU with no references, we will append them incrementally
        self.bleu = BLEU([], self.weights)
        # In the optimized version, the C++ object holds the references, 
        # so we don't need to keep a separate list in Python.
        # We will use a counter for the number of references.
        self.num_references = 0

    def append_reference(self, ref: Union[str, List[str]]):
        if isinstance(ref, list):
            for r in ref:
                self.bleu.append_reference(self.tokenizer(r))
                self.num_references += 1
        else:
            self.bleu.append_reference(self.tokenizer(ref))
            self.num_references += 1

    def __call__(self, hypotheses: List[str]):
        start_time = time()
        
        tokenized_hypotheses = list(map(self.tokenizer, hypotheses))
        scores = list(self.bleu.get_score(tokenized_hypotheses).values())
        stop_time = time()-start_time
        
        print(f"num_refs={self.num_references}, {start_time-self.time_last_run=}, {stop_time=}")
        self.time_last_run = time()
        return np.asarray(scores).mean(axis=0)

# --- Example Usage ---
if __name__ == '__main__':
    references = [
        "The cat sat on the mat. A feline was resting on the rug. On the mat, a cat was sitting."
    ] * 90000
    hypotheses = [
        "The cat is on the mat.",
        "A cat sat on the mat.",
    ]
    
    print(f"Calculating BLEU for {len(hypotheses)} hypotheses.")

    # --- Test Sequential Implementation ---
    bleu_reward_seq = SelfBleuReward(grams=[2, 4])
    bleu_reward_seq.append_reference(references[:2])
    sequential_scores = bleu_reward_seq(hypotheses)
    print(f"Sequential Scores (2-gram, 4-gram): {sequential_scores}")
    bleu_reward_seq.append_reference(references)
    sequential_scores = bleu_reward_seq(hypotheses)
    print(f"Sequential Scores (2-gram, 4-gram): {sequential_scores}")
    bleu_reward_seq.append_reference(references)
    sequential_scores = bleu_reward_seq(hypotheses)
    print(f"Sequential Scores (2-gram, 4-gram): {sequential_scores}")

    print("looking for segfaults (no references, no hypotheses)")
    bleu_reward_seq = SelfBleuReward(grams=[2, 4])
    sequential_scores = bleu_reward_seq(hypotheses)
    print(f"Sequential Scores (2-gram, 4-gram): {sequential_scores}")
    sequential_scores = bleu_reward_seq([])
    print(f"Sequential Scores (2-gram, 4-gram): {sequential_scores}")
