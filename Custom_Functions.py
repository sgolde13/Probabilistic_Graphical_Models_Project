#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For the EN.625.692.81.SP24 Probabilistic Models Semester Project
By Shelby Golden April 27th, 2024

This file contains all of the custom functions used in the Analysis Script.py
file. They are separated for clarity.

Enviroment: conda activate "/Users/shelbygolden/miniconda3/envs/spyder-env" 
Spyder: /Users/shelbygolden/miniconda3/envs/spyder-env/bin/python
"""
from itertools import compress
from itertools import groupby
from Bio import SeqIO
from Bio import Align
    # Install BioPython https://biopython.org/wiki/Packages
    # https://biopython.org/docs/1.75/api/Bio.Seq.html
from scipy.stats import laplace_asymmetric
import numpy as np


def all_equal(iterable):
    """
    Checks that all elements in a list are the same
    If all elements are equal to True, then this passes the check.
    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def import_seq(file_location, file_type = "fasta"):
    """
    Extracts the raw protein primary sequence 
    from the PBD database file.
    """
    for seq_record in SeqIO.parse(file_location, file_type):
        return str(seq_record.seq) #print(repr(seq_record.seq))
    
    
def import_txt(file_path):
    """
    Sreamlines importing text files
    """
    f = open(file_path, 'r')
    content = f.read()
    
    return content



def replace_str(list_remove, protein_seq, replace_with):
    """
    Replace protein sequence items from a list
    """
    for char in list_remove:
        protein_seq = protein_seq.replace(char, replace_with)
    
    return protein_seq



def regex_aa(list_change, protein_seq):
    """
    For 4.3. Feature extraction a. Regular expression template in the topic
    paper used for this analysis.
   
    Convert a protein primary sequence of amino acids into the
    regex template
    """
    for residue in list_change:
    
        if residue == 'hydrophobic':
            protein_seq = replace_str(list_change[residue], protein_seq, 'h')
         
        elif residue == 'ionisable':
            protein_seq = replace_str(list_change[residue], protein_seq, 'i')
            
        elif residue == 'other':
            """
            From Todor
            https://stackoverflow.com/questions/12897374/get-unique-values-from-a-list-in-python
            """
            used = set()
            mylist = list(protein_seq)
            unique = [x for x in mylist if x not in used and (used.add(x) or True)]
            
            
            not_changed = [x for x in unique if (x != 'h') & (x != 'i')]
            #replace_x  = ''.join(str(x) for x in not_changed)
            
            
            protein_seq = replace_str(not_changed, protein_seq, 'x')
            
    return protein_seq.upper()


def parse_protein(protein, protein_ps, spacing, segmenting):
    """
    For 4.1. Protein structural graph in the topic paper used for this analysis.
    
    Create candidate, plausible sequence parsing for the normal amino acid
    sequence and the PSIPRED score.
    """
    
    if (len(protein) == len(protein_ps)) == False:
        # the lengths need to be the same for the function to work
        return print('lengths dont match')
        
    
    fin_segments = {}
    fin_segments_ps = {}
    for j in range(len(spacing)):
    
        helix_len = segmenting["s-B23"] + segmenting["s-T3"][j] + segmenting["s-B1"] + segmenting["s-T1"][j]
        num_helix = (len(protein) - 2*segmenting["s-I"]) // helix_len
    
    
        # split the remainder of non-helix for s-I at the start and end of the fold
        first_sI = (len(protein) - (helix_len * num_helix)) // 2
        
        
        whole_segments = []
        whole_segments_ps = []
        
        
        # first s-I segment
        whole_segments.append(protein[0:first_sI])
        whole_segments_ps.append(protein_ps[0:first_sI])
        
        
        for i in range(num_helix):
            if i == 0:
                start = first_sI + 1
        
            
            whole_segments.append(protein[start:start + segmenting["s-B23"]])
            whole_segments_ps.append(protein_ps[start:start + segmenting["s-B23"]])
            next_start = start + segmenting["s-B23"]
        
    
            whole_segments.append(protein[next_start:next_start + segmenting["s-T3"][j]])
            whole_segments_ps.append(protein_ps[next_start:next_start + segmenting["s-T3"][j]])
            next_start = next_start + segmenting["s-T3"][j]
    
        
            whole_segments.append(protein[next_start:next_start + segmenting["s-B1"]])
            whole_segments_ps.append(protein_ps[next_start:next_start + segmenting["s-B1"]])
            next_start = next_start + segmenting["s-B1"]
        
            # reset the start
            whole_segments.append(protein[next_start:next_start + segmenting["s-T1"][j]])
            whole_segments_ps.append(protein_ps[next_start:next_start + segmenting["s-T1"][j]])
            start = next_start + segmenting["s-T1"][j]
    
            
        # add the last s-I
        whole_segments.append(protein[start:len(protein)])
        whole_segments_ps.append(protein_ps[start:len(protein)])
        
        
        del start
        fin_segments[j] = whole_segments
        fin_segments_ps[j] = whole_segments_ps
        
    return fin_segments, fin_segments_ps



def regex_all_seg(protein_seg, template):
    """
    For 4.3. Feature extraction a. Regular expression template in the topic
    paper used for this analysis.
    
    Convert all list elements to the regex form
    """
    reg_protein_seg = []
    for i in range(len(protein_seg)):
        reg_protein_seg.append([regex_aa(template, x) for x in protein_seg[i]])
    
    return reg_protein_seg



def check(test_str, accepted):
    """
    Confirm that the PSIPRED score matches the expected inclusion of strings
    criteria that represent residues with predicted folding features.
    
    Searches string to confirm that all entries only have the expected characters.
    https://stackoverflow.com/questions/1323364/in-python-how-to-check-if-a-string-only-contains-certain-characters
    """
    boolian = set(test_str) <= accepted
    return boolian



def check_alignment(reg_exp_parsed, target_seq):
    """
    For 4.3. Feature extraction a. Regular expression template in the topic
    paper used for this analysis.
    
    B2_T2_B2 = 'HXHXXIXHX':
    The target sequence for B2-T2-B2 is 9 bp long. Segments of regex amino
    acid sequences are separated into those that are [6, 10] bp long, to
    build in a little flexability for insertions and deletions. More
    tolerance for insertions than deletions.
    
    Match scores are then counted for [6, 9] matches and this is considered
    "match = 1". All other are classified as "match = 0". Again, this builds
    tolerance for matches between 6 and at most 9 bp.
    
    
    B1 = 'HXH':
    The target sequence for B1 is 3 bp long. Segments of regex amino
    acid sequences are separated into those that are [2, 5] bp long, to
    build in a little flexability for insertions and deletions. More
    tolerance for insertions than deletions.
    
    Match scores are then counted for [2, 3] matches and this is considered
    "match = 1". All other are classified as "match = 0". Again, this builds
    tolerance for matches between 2 and at most 3 bp.
    """
    B2_T2_B2 = 'HXHXXIXHX'
    B1 = 'HXH'
    
    aligner = Align.PairwiseAligner(match_score=1.0)

    align_score = []
    for i in range(len(reg_exp_parsed)):
        candidate_seg = reg_exp_parsed[i]
        
        
        # start with a list of "not matched" outcomes. Will replace values
        # based on the following boolean
        candidate_score = [0] * len(candidate_seg)
        
        
        # get the alignment scores by the segment for the two possible regex expressions
        if target_seq == B2_T2_B2:
            # the sequence must be at least 6 bp long, but we add flexibility
            # for some variation up to 2 additional bp. This finds which
            # segments in a candidate parsing fits these parameters
            fit_len = [(y >= 6) & (y <= 10) for y in [len(x) for x in candidate_seg ]]
            list_el_fit = list(compress(candidate_seg, fit_len))
            
            index = [i for i, x in enumerate(fit_len) if x == True]
            

            for j in range(len(list_el_fit)):
                target = target_seq
                query = list_el_fit[j]
                score = aligner.score(target, query)
                
                
                # again we add some flexibility. If the score indicates at least
                # 6 of the 9 potential matching then this is sufficient
                if (score >= 6) & (score <= 9):
                    candidate_score[index[j]] = 1
            
            
            align_score.append(candidate_score)
            
            
        elif target_seq == B1:
            # the sequence must be at least 3 bp long, but we add flexibility
            # for some variation up to 2 additional bp. This finds which
            # segments in a candidate parsing fits these parameters
            fit_len = [(y >= 2) & (y <= 5) for y in [len(x) for x in candidate_seg ]]
            list_el_fit = list(compress(candidate_seg, fit_len))
            
            index = [i for i, x in enumerate(fit_len) if x == True]
            

            for j in range(len(list_el_fit)):
                target = target_seq
                query = list_el_fit[j]
                score = aligner.score(target, query)
                
                
                # again we add some flexibility. If the score indicates at least
                # 6 of the 9 potential matching then this is sufficient
                if (score >= 2) & (score <= 3):
                    candidate_score[index[j]] = 1
            
            
            align_score.append(candidate_score)  
            
            
        # check if there is alignment inside of outside the expected segment
        in_set = []
        out_set = []
        for j in range(len(align_score)):
            seq_parsing = align_score[j]
            
            # strip the s-I head and tail sequence so that the index associated
            # with s-T1 and s-T3 can be found by the predictable patern
            fold_only = seq_parsing[1:len(seq_parsing)-1]
            freq = int(len(fold_only) / 4)
            
            patern = ['s-B23', 's-T3', 's-B1', 's-T1'] * freq  
            
            
            
            if target_seq == B2_T2_B2:
                index = [i for i, x in enumerate(patern) if x == 's-B23']
                anti_index = [i for i, x in enumerate(patern) if x != 's-B23']
                
            elif target_seq == B1:
                index = [i for i, x in enumerate(patern) if x == 's-B1']  
                anti_index = [i for i, x in enumerate(patern) if x != 's-B1']
                
            
            in_set.append( sum( [seq_parsing[index[x]] for x in range(len(index))] ) )
            out_set.append( sum( [seq_parsing[anti_index[x]] for x in range(len(anti_index))] ) )
            
            
        
    return in_set, out_set



def secondary_scores(psipred_seq, whole_ps_seq):
    """
    For 4.3. Feature extraction c. Secondary structure prediction scores in 
    the topic paper used for this analysis.
    
    Using the color coding grey (g) = coil, yellow (y) = sheet, and pink (p)
    = helix, the PSIPRED were converted into a string. The mean of simple boolean
    adherence to these codings is used to determine the score for each
    folding motif prediction in that sequence segment.
    """
    results = []
    
    whole_seq = []
    # Portion of the function for the whole sequnce
    helix1 = sum( [int(x == 'p') for x in whole_ps_seq ])
    sheet1 = sum( [int(x == 'y') for x in whole_ps_seq ])
    coil1 = sum( [int(x == 'g') for x in whole_ps_seq ])
    length1 = len(whole_ps_seq)

    probability = [ round(x / length1, 3) for x in [helix1, sheet1, coil1] ]
    
    
    whole_seq.append(probability)
    
    # the whole sequence has no features associated with specific segments
    # since all are considered s-I
    i = 1
    while i < 6:
        whole_seq.append([None, None, None])
        i += 1
    
    # Portion of the function for the parsed sequence
    helix_all = []
    sheet_all = []
    coil_all = []
    length_all = []
    for i in range(len(psipred_seq)):
        segment = psipred_seq[i]
        
        
        helix = []
        sheet = []
        coil = []
        length = []
        for j in range(len(segment)):
            helix.append( sum( [int(x == 'p') for x in segment[j] ]) )
            sheet.append( sum([int(x == 'y') for x in segment[j] ]) )
            coil.append( sum([int(x == 'g') for x in segment[j] ]) )
            
            length.append( len(segment[j]) )
            
            
        helix_all.append(helix)
        sheet_all.append(sheet)
        coil_all.append(coil)
        length_all.append(length)
        
        
    for k in range(len(psipred_seq)):
        segment = psipred_seq[k]
        
        helix_seg = helix_all[k]
        sheet_seg = sheet_all[k]
        coil_seg = coil_all[k]
        length_seg = length_all[k]
        
        
        # strip the s-I head and tail sequence so that the index associated
        # with s-T1 and s-T3 can be found by the predictable patern
        fold_only = segment[1:len(segment)-1]
        freq = int(len(fold_only) / 4)
        
        unique_patern = ['s-B23', 's-T3', 's-B1', 's-T1']
        patern = unique_patern * freq
        
        
        cal_mean = []
        # add the whole sequence to each instantiation
        cal_mean.append(probability)
        
        # add the folding region
        cal_mean.append([ round(x / sum(length_seg), 3) for x in [sum(helix_seg), sum(sheet_seg), sum(coil_seg)] ])
        
        for j in range(len(unique_patern)):
            index = [i for i, x in enumerate(patern) if x == unique_patern[j] ]
    
            denoinator = sum( [ length_seg[index[x]] for x in range(len(index))] )
            hel_num = sum( [ helix_seg[index[x]] for x in range(len(index))] )
            she_num = sum( [ sheet_seg[index[x]] for x in range(len(index))] )
            coi_num = sum( [ coil_seg[index[x]] for x in range(len(index))] )
            
            cal_mean.append([ round(x / denoinator, 3) for x in [hel_num, she_num, coi_num] ])
        
        # for each candidate segmentation, the mean helix, sheet, and coil is 
        # calculated for the ['whole seq', 's-B23', 's-T3', 's-B1', 's-T1'] paterns.
        results.append(cal_mean)
        
        
    results.append(whole_seq)
            
    return results



def laplace_prob(sequence):
    """
    For 4.3. Feature extraction d. Segment length in the topic paper used 
    for this analysis.
    
    The Asymmetric Laplace density probability associated with the length of
    a sequence for the s-T1 and s-T3 segments is calculated. The parameters
    for the density (kappa, location, scale) were not provided in the paper,
    and so they were guessed by trial-and-error.
    """
    
    T1_all = []
    T3_all = []
    for i in range(len(sequence)):
        seq_parsing = sequence[i]
        
        # strip the s-I head and tail sequence so that the index associated
        # with s-T1 and s-T3 can be found by the predictable patern
        fold_only = seq_parsing[1:len(seq_parsing)-1]
        freq = int(len(fold_only) / 4)
        
        patern = ['s-B23', 's-T3', 's-B1', 's-T1'] * freq    
        
        index_T3 = [i for i, x in enumerate(patern) if x == 's-T3']
        index_T1 = [i for i, x in enumerate(patern) if x == 's-T1']
        
        
        T1_score = []
        T3_score = []
        for j in range(len(index_T3)):
            t1_len = len(fold_only[index_T1[j]])
            t3_len = len(fold_only[index_T3[j]])
            
            T1_score.append( laplace_asymmetric.pdf(t1_len, kappa=0.5, loc=5, scale=0.6) )
            T3_score.append( laplace_asymmetric.pdf(t3_len, kappa=0.6, loc=5, scale=1) )
            
            
        # all spacings within one candidate segmentation should be the same. This
        # checks that that assumption is true.
        if (all_equal(T1_score) == True) and (all_equal(T3_score) == True):
            T1_all.append(T1_score[0])
            T3_all.append(T3_score[0])
            
        else:
            return print('segmentation spacing is not all equal within a parsing')
        
    return T1_all, T3_all



def dist_B23(sequence):
    """
    For 4.3. Feature extraction c. Distance between adjacent s-B23 segments
    
    Because of how segmentation is done, the distance between s-B23 segments
    should be the same. Then the mean-centered and normalized average
    distance can be one value for each potential segmentation. Notice
    that the x - mu expression is not an absolute value. This allows us
    to know segments that has a lower than average distance.
    """
    
    seg_dist = []
    for i in range(len(sequence)):
        seq_parsing = sequence[i]
        
        # strip the s-I head and tail sequence so that the index associated
        # with s-T1 and s-T3 can be found by the predictable patern
        fold_only = seq_parsing[1:len(seq_parsing)-1]
        freq = int(len(fold_only) / 4)
        
        patern = ['s-B23', 's-T3', 's-B1', 's-T1'] * freq    
        
        
        index_T3 = [i for i, x in enumerate(patern) if x == 's-T3']
        index_T1 = [i for i, x in enumerate(patern) if x == 's-T1']
        
        distance = []
        for j in range(freq):
            aa_between = ''.join(fold_only[index_T3[j]:index_T1[j] + 1]) 
            distance.append(len( aa_between ))
        
        
        seg_dist.append(distance)
        
    # all spacings within one candidate segmentation should be the same. This
    # checks that that assumption is true.
    if(all([all_equal(x) for x in seg_dist]) == True):
        all_spacing = [x[0] for x in seg_dist]
        
        mu = np.mean( all_spacing )
        sigma = np.std( all_spacing )
        
        return [round((x - mu)/sigma, 3) for x in all_spacing]
        
        
    else:
        return print('segmentation spacing is not all equal within a parsing')

