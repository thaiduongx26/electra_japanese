# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenization classes, the same as used for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import six
import tensorflow.compat.v1 as tf
import sentencepiece as sp



def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token, _ = token.split("\t")
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


# def convert_by_vocab(vocab, items):
#     """Converts a sequence of [tokens|ids] using the vocab."""
#     output = []
#     for item in items:
#         output.append(vocab[item])
#     return output


# def convert_tokens_to_ids(vocab, tokens):
#     return convert_by_vocab(vocab, tokens)


# def convert_ids_to_tokens(inv_vocab, ids):
#     return convert_by_vocab(inv_vocab, ids)

def convert_by_vocab(vocab, items, unk_info):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        if item in vocab:
            output.append(vocab[item])
        else:
            output.append(unk_info)
    return output


def convert_tokens_to_ids(vocab, tokens):
    """Id of <unk> is assumed as 0 accroding to sentencepiece"""
    return convert_by_vocab(vocab, tokens, unk_info=0)


def convert_ids_to_tokens(inv_vocab, ids):
    """Token of unknown word is assumed as <unk> according to sentencepiece"""
    return convert_by_vocab(inv_vocab, ids, unk_info="<unk>")

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


# class FullTokenizer(object):
#     """Runs end-to-end tokenziation."""

#     def __init__(self, vocab_file, do_lower_case=True):
#         self.vocab = load_vocab(vocab_file)
#         self.inv_vocab = {v: k for k, v in self.vocab.items()}
#         self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
#         self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

#     def tokenize(self, text):
#         split_tokens = []
#         for token in self.basic_tokenizer.tokenize(text):
#             for sub_token in self.wordpiece_tokenizer.tokenize(token):
#                 split_tokens.append(sub_token)

#         return split_tokens

#     def convert_tokens_to_ids(self, tokens):
#         return convert_by_vocab(self.vocab, tokens)

#     def convert_ids_to_tokens(self, ids):
#         return convert_by_vocab(self.inv_vocab, ids)

class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, model_file='', do_lower_case=True):
        print('Loading sentencepiece model from: ', model_file)
        self.tokenizer = SentencePieceTokenizer(model_file, do_lower_case=do_lower_case)
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        split_tokens = self.tokenizer.tokenize(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Id of <unk> is assumed as 0 accroding to sentencepiece"""
        return convert_by_vocab(self.vocab, tokens, unk_info=0)

    def convert_ids_to_tokens(self, ids):
        """Token of unknown word is assumed as <unk> according to sentencepiece"""
        return convert_by_vocab(self.inv_vocab, ids, unk_info="<unk>")

class SentencePieceTokenizer(object):
    """Runs SentencePiece tokenization (from raw text to tokens list)"""

    def __init__(self, model_file=None, do_lower_case=True):
        """Constructs a SentencePieceTokenizer."""
        if model_file == '' or model_file == None:
            print('Empty sentence piece model path')
            model_file = 'model_sentence_piece/wiki-ja.model'
            print('Loading sentence piece from specifix path: ', model_file)
        self.tokenizer = sp.SentencePieceProcessor()
        if not self.tokenizer.Load(model_file):
            #print("Loaded a trained SentencePiece model.")
        #else:
            print("You have to give a path of trained SentencePiece model.")
            sys.exit(1)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        if self.do_lower_case:
            text = text.lower()
        output_tokens = self.tokenizer.EncodeAsPieces(text)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
