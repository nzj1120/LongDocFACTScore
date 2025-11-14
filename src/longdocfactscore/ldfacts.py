# Suppress annoying warnings from this issue which cannot be solved: https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md and transformers packages
import warnings
warnings.filterwarnings("ignore")

from typing import List, Optional

import torch
import torch.nn as nn
import traceback
from transformers import AutoTokenizer, BartForConditionalGeneration
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util

def _auto_select_device(requested_device: Optional[str]) -> str:
    """Return a valid torch device string."""

    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _split_into_sentences(text: str) -> List[str]:
    """Light-weight sentence splitter that supports both Chinese and English text."""

    if not text:
        return []

    # Normalise whitespace so that regex splitting behaves well.
    normalised = re.sub(r"\s+", " ", text.strip())
    # Split on common Chinese and English sentence delimiters while keeping the delimiter
    # attached to the sentence it terminates.
    parts = re.split(r"(?<=[。！？!?])", normalised)
    sentences: List[str] = []
    buffer = ""
    for part in parts:
        if not part:
            continue
        buffer += part
        if re.search(r"[。！？!?]$", part):
            cleaned = buffer.strip()
            if cleaned:
                sentences.append(cleaned)
            buffer = ""
    if buffer.strip():
        sentences.append(buffer.strip())
    return sentences


class LongDocFACTScore():
    def __init__(
        self,
        device: Optional[str] = None,
        model: str = "BARTScore",
        sent_model_name_or_path: str = "uer/sbert-base-chinese-nli",
        bart_model_name_or_path: str = "fnlp/bart-large-chinese",
        bart_tokenizer_name_or_path: Optional[str] = None,
    ):
        resolved_device = _auto_select_device(device)
        self.sent_model = SentenceTransformer(sent_model_name_or_path)
        self.sent_model.to(resolved_device)
        self.device = resolved_device
        if model == "BARTScore":
            self.metric = BARTScore(
                device=self.device,
                checkpoint=bart_model_name_or_path,
                tokenizer_name_or_path=bart_tokenizer_name_or_path,
            )
            self.metric_function = self.metric.bart_score
        else:
            raise ValueError("LongDocFACTScore currently only supports BARTScore")


    def get_surrounding_sentences(self, sentence_array, ii):
        if ii > 0 and ii < len(sentence_array) - 1:
            sents = " ".join(np.array(sentence_array)[ii - 1 : ii + 1])
        elif ii == 0:
            sents = " ".join(np.array(sentence_array)[:2])
        elif ii == len(sentence_array) - 1:
            sents = " ".join(np.array(sentence_array)[ii - 1 :])
        return sents

    def group_into_sections(self, sentence_array, num_sent):
        sectioned_sents = []
        for ii in range(0, len(sentence_array), num_sent):
            sectioned_sents.append(" ".join(sentence_array)[ii : ii + num_sent])
        return sectioned_sents

    def score_src_hyp_long(self, srcs, hyps):
        all_scores = []
        # src is a list containing source documents.
        # hyps is a list containing predicted documents
        for src, hyp in zip(srcs, hyps):
            src_sents = _split_into_sentences(src)
            if not src_sents:
                raise ValueError(
                    "The source document does not contain any detectable sentences."
                )
            sentence_embeddings_src = self.sent_model.encode(
                src_sents, show_progress_bar=False
            )
            doc_scores = []
            hyp_array = _split_into_sentences(hyp)
            if not hyp_array:
                raise ValueError(
                    "The hypothesis document does not contain any detectable sentences."
                )
            for idx, hyp_sentence in enumerate(hyp_array):
                # for each sentence in summary, calculate the most similar sentence in the source article
                sentence_embeddings_hyp = self.sent_model.encode(
                    [hyp_sentence], show_progress_bar=False
                )
                scores = util.cos_sim(sentence_embeddings_hyp, sentence_embeddings_src)[
                    0
                ]
                sorted_idxs = np.argsort(-1 * scores)
                similar_src_sentences = []
                #  get sentences surrounding the most similar sentences in the source article
                for ii in sorted_idxs[0:3]:
                    similar_sents = self.get_surrounding_sentences(src_sents, ii)
                    similar_src_sentences.append(similar_sents)
                # calculate metric for 3 most similar sections of source article
                scores = self.metric_function(
                    similar_src_sentences,
                    [hyp_sentence for i in range(0, len(similar_src_sentences))],
                )
                # Take the max scoring section to use
                score = np.max(scores)
                doc_scores.append(score)

            # calculate average score over whole doc
            doc_score = np.mean(doc_scores)
            all_scores.append(doc_score)
        return all_scores


# code taken from https://github.com/neulab/BARTScore/blob/main/bart_score.py
class BARTScore():
    def __init__(
        self,
        device: str = "cuda:0",
        checkpoint: str = "facebook/bart-large",
        tokenizer_name_or_path: Optional[str] = None,
    ):
        # Set up model
        self.device = device
        self.max_length = 1024
        tokenizer_source = tokenizer_name_or_path or checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(
            reduction="none", ignore_index=self.model.config.pad_token_id
        )
        self.lsm = nn.LogSoftmax(dim=1)
        

    def bart_score(self, srcs, tgts, batch_size=4):
        ### Taken from 
        """Score a batch of examples"""
        score_list = []

        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i : i + batch_size]
            tgt_list = tgts[i : i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    src_tokens = encoded_src["input_ids"].to(self.device)
                    src_mask = encoded_src["attention_mask"].to(self.device)

                    tgt_tokens = encoded_tgt["input_ids"].to(self.device)
                    tgt_mask = encoded_tgt["attention_mask"]
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens, attention_mask=src_mask, labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f"source: {src_list}")
                print(f"target: {tgt_list}")
                exit(0)
        return score_list
