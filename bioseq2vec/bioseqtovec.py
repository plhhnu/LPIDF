from numpy import shape
import numpy as np
import sys
from bioseq2vec import Seq2VecR2R
from bioseq2vec.util import DataGenterator
import argparse
import pandas as pd

def read_fasta_file(fasta_file):
    seq_dict = {}
    with open(fasta_file, 'r') as fp:
        name = ''
        for line in fp:
            # let's discard the newline at the end (if any)
            line = line.rstrip()
            # distinguish header from sequence
            if line[0] == '>':  # or line.startswith('>')
                # it is the header
                name = line[1:]  # discarding the initial >
                seq_dict[name] = ''
            else:
                # it is sequence
                seq_dict[name] = seq_dict[name] + line.upper()
    return seq_dict

def get_4_trids():
    '''
    Returns: List of all 4-mer nucleic acid combinations of RNA, e.g. [AAAA,AAAC,AAAG，......TTTG, TTTT]
    -------
    '''

    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars) ** 4
    for i in range(0, end):
        n = i
        ch0 = chars[int(n % base)]
        n = n / base
        ch1 = chars[int(n % base)]
        n = n / base
        ch2 = chars[int(n % base)]
        n = n / base
        ch3 = chars[int(n % base)]
        nucle_com.append(ch0 + ch1 + ch2 + ch3)
    return nucle_com


def get_k_nucleotide_composition(tris, seq):
    '''
    Parameters
    ----------
    tris: List of all possible mers
    seq: input single sequence

    Returns: kmer feature of single sequence
    -------
    '''

    seq_len = len(seq)
    tri_feature = []
    k = len(tris[0])
    tmp_fea = [0] * len(tris)
    for x in range(len(seq) + 1 - k):
        kmer = seq[x:x + k]
        if kmer in tris:
            ind = tris.index(kmer)
            tmp_fea[ind] = tmp_fea[ind] + 1
    tri_feature = [float(val) / seq_len for val in tmp_fea]
    # pdb.set_trace()
    return tri_feature


def get_words(k, seq):
    seq_len = len(seq)
    words = []

    # tmp_fea = [0] * len(tris)
    # i = 0
    # while len(seq) - i > k:
    #     word = seq[i:i + k]
    #     words.append(str(word))
    #     i = i + k
    for x in range(len(seq) + 1 - k):
        kmer = seq[x:x + k]
        words.append(str(kmer))
    # tri_feature = [float(val)/seq_len for val in tmp_fea]
    # pdb.set_trace()
    return words



if __name__ == '__main__':

    RNA_seq_dict = {}
    RNA_feature = []
    RNA_seqs = read_fasta_file('./data/LncRNA Sequence.txt')
    RNA_seqs_names = pd.read_csv('./data/LncRNA.csv')['lncRNA_id']
    tris = get_4_trids()
    for RNA in RNA_seqs_names:  # iteritems() removed in python 3
        if RNA in RNA_seqs:
            RNA_seq = RNA_seqs[RNA]
            # k-mer feature
            RNA_tri_fea = get_k_nucleotide_composition(tris, RNA_seq)
            RNA_feature.append(RNA_tri_fea)
    RNA_feature = np.array(RNA_feature)
    np.savetxt('./RNA.txt', RNA_feature)

    protein_seq_dict = {}
    pro_feature = []
    pro_seqs = read_fasta_file('./data/Protein Sequence.txt')
    pro_seqs_names = pd.read_csv('./data/Protein.csv')['protein_id']
    bioseq2vec_pro = Seq2VecR2R()
    # word-level pretrained models
    bioseq2vec_pro.load_customed_model("pretrained models/seq2vec_protein_word.model")
    for pro in pro_seqs_names:  # iteritems() removed in python 3
        if pro in pro_seqs:
            # transform sequences
            bioseq2vec_pro_feature = bioseq2vec_pro.transform([get_words(3, pro)]).reshape(-1)
            pro_feature.append(bioseq2vec_pro_feature)
    pro_feature = np.array(pro_feature)
    np.savetxt('./pro.txt', pro_feature)
