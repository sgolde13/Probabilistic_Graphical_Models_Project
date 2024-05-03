# -*- coding: utf-8 -*-
"""
For the EN.625.692.81.SP24 Probabilistic Models Semester Project
By Shelby Golden April 27th, 2024

This is the primary analysis script used for the paper

Enviroment: conda activate "/Users/shelbygolden/miniconda3/envs/spyder-env" 
Spyder: /Users/shelbygolden/miniconda3/envs/spyder-env/bin/python
"""

##########################################################
##########################################################
## package import and function definitions


# Some of the functions are called from the local copy of the GitHub 
# repo: https://github.com/alrojo/biRNN-CRF
# Add the importation path with the following code if it is missing
import sys
print(sys.path)

sys.path.append('/Users/shelbygolden/Desktop/Life/Personal/College/Masters/Johns Hopkins/9.2024 Spring_Probabilistic Graphical Models/Semester Project/Probabilistic_Graphical_Models_Project')


# Check the current working directory
import os
current_working_directory = os.getcwd()
print(current_working_directory)

os.chdir('/Users/shelbygolden/Desktop/Life/Personal/College/Masters/Johns Hopkins/9.2024 Spring_Probabilistic Graphical Models/Semester Project/Probabilistic_Graphical_Models_Project')



# Packages for the rest of the script
import numpy as np
from Bio import SeqIO
    # Install BioPython https://biopython.org/wiki/Packages
    # https://biopython.org/docs/1.75/api/Bio.Seq.html
import random
import pyhmmer



# functions for the rest of the script
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
    Part of the Node Features section
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
            mylist = list(pdb_1DBG)
            unique = [x for x in mylist if x not in used and (used.add(x) or True)]
            
            
            not_changed = [x for x in unique if (x != 'h') & (x != 'i')]
            #replace_x  = ''.join(str(x) for x in not_changed)
            
            
            protein_seq = replace_str(not_changed, protein_seq, 'x')
            
    return protein_seq.upper()



def parse_protein(protein):
    """
    Create candidate, plausible sequence parsing based on the 4.1. Protein structural
    graph for β-helix in the topic paper Liu et al 2006
    """
    fin_segments = {}
    for j in range(len(spacing)):
    
        helix_len = segmenting["s-B23"] + segmenting["s-T3"][j] + segmenting["s-B1"] + segmenting["s-T1"][j]
        num_helix = (len(protein) - 2*segmenting["s-I"]) // helix_len
    
    
        # split the remainder of non-helix for s-I at the start and end of the fold
        first_sI = (len(protein) - (helix_len * num_helix)) // 2
        
        
        whole_segments = []
        
        # first s-I segment
        whole_segments.append(protein[0:first_sI])
        
        
        for i in range(num_helix):
            if i == 0:
                start = first_sI + 1
        
            
            whole_segments.append(protein[start:start + segmenting["s-B23"]])
            next_start = start + segmenting["s-B23"]
        
    
            whole_segments.append(protein[next_start:next_start + segmenting["s-T3"][j]])
            next_start = next_start + segmenting["s-T3"][j]
    
        
            whole_segments.append(protein[next_start:next_start + segmenting["s-B1"]])
            next_start = next_start + segmenting["s-B1"]
        
            # reset the start
            whole_segments.append(protein[next_start:next_start + segmenting["s-T1"][j]])
            start = next_start + segmenting["s-T1"][j]
    
            
        # add the last s-I
        whole_segments.append(protein[start:len(protein)])
        
        
        del start
        fin_segments[j] = whole_segments
        
    return fin_segments



def regex_all_seg(protein_seg):
    """
    Convert all list elements to the regex form
    """
    reg_protein_seg = []
    for i in range(len(protein_seg)):
        reg_protein_seg.append([regex_aa(reg_exp_template, x) for x in protein_seg[i]])
    
    return reg_protein_seg



##########################################################
##########################################################
## import sequences and segmentation

##########################################################
## protein primary sequence

# Known to have a right-handed beta-helix
pdb_1DBG = import_seq("PDB FASTA Files/rcsb_pdb_1DBG.fasta")
pdb_1DAB = import_seq("PDB FASTA Files/rcsb_pdb_1DAB.fasta")
pdb_1EA0 = import_seq("PDB FASTA Files/rcsb_pdb_1EA0.fasta")
pdb_1QJV = import_seq("PDB FASTA Files/rcsb_pdb_1QJV.fasta")
pdb_1TYU = import_seq("PDB FASTA Files/rcsb_pdb_1TYU.fasta")


# Predicted to have a beta-helix by UNIPROT
Q6LZ14 = import_seq("PDB FASTA Files/UNIPROT_Q6LZ14.fasta")
P35338 = import_seq("PDB FASTA Files/UNIPROT_P35338.fasta")
Q6ZGA1 = import_seq("PDB FASTA Files/UNIPROT_Q6ZGA1.fasta")



##########################################################
## secondary structure prediction scores
"""
g: light grey = coil
y: yellow = strand
p: pink = helix
u: no color = undefined
"""

# Known to have a right-handed beta-helix
pdb_1DBG_ps = import_txt("PSIPRED Annotation/pdb_1DBG_PSIPRED Annotation.txt")
pdb_1DAB_ps = import_txt("PSIPRED Annotation/pdb_1DAB_PSIPRED Annotation.txt")
pdb_1EA0_ps = import_txt("PSIPRED Annotation/pdb_1EA0_PSIPRED Annotation.txt")
pdb_1QJV_ps = import_txt("PSIPRED Annotation/pdb_1QJV_PSIPRED Annotation.txt")
pdb_1TYU_ps = import_txt("PSIPRED Annotation/pdb_1TYU_PSIPRED Annotation.txt")


# Predicted to have a beta-helix by UNIPROT
Q6LZ14_ps = import_txt('PSIPRED Annotation/Q6LZ14_PSIPRED Annotation.txt')
P35338_ps = import_txt('PSIPRED Annotation/P35338_PSIPRED Annotation.txt')




##########################################################
## segmentation
"""
Section 4.1. Protein structural graph for β-helix in the topic paper 
Liu et al 2006 has notations about restrictions for the length for some of the
components for the fold segments. Arbitrarily, s-I is set to around 10
to allow for some preceeding and ending s-I tails on the sequence folding.


Patern: SI-B23-T3-B1-T1  -  B23-T3-B1-T1  -  ...  -  T3-B1-T1-B23-SI

s-B23 and s-B1 lengths are fixed as 8 and 3 respectively for two reasons:
    1. these are the numbers of residues shared by all known β-helices
    2. it helps limit the search space and reduce the computational costs
    
s-T3 and s-T1 connect s-B23 and s-B1 and range from 1 to 80 amino acids. 
NOTE: β-helix structures will break if the insertion is too long

State s-I is the non-β-helix state at the start and end of the folding patern.

A protein w/o any β-helix structures is defined to be one single s-I node.
"""


spacing = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]


segmenting = {'s-B23': 8, 's-T3': random.sample(spacing, len(spacing)), 
              's-B1': 3, 's-T1': random.sample(spacing, len(spacing)), 's-I': 10}

    

parsed = parse_protein(pdb_1DBG)
    

##########################################################
##########################################################
## node feature extraction


##########################################################
## regular expression template

B2_T2_B2 = 'HXHXXIXHX'
    
reg_exp_template = {'hydrophobic': ['A', 'F', 'I', 'L', 'M', 'V', 'W', 'Y'], 
                    'ionisable': ['D', 'E', 'R', 'K'],
                    'other': {}}



regex_aa(reg_exp_template, parsed[1])


reg_parsed = regex_all_seg(parsed)

    


##########################################################
## probabilistic HMM profiles

alphabet = pyhmmer.easel.Alphabet.amino()
seq1 = pyhmmer.easel.TextSequence(name=b"seq1", sequence="LAGYLLPIMVLLNVAPC")
seq2 = pyhmmer.easel.TextSequence(name=b"seq2", sequence="LAGY----MVLLNLAGC")
msa  = pyhmmer.easel.TextMSA(name=b"msa", sequences=[seq1, seq2])

msa_d = msa.digitize(alphabet)

builder = pyhmmer.plan7.Builder(alphabet)
background = pyhmmer.plan7.Background(alphabet)
hmm, _, _ = builder.build_msa(msa_d, background)


pipeline = pyhmmer.plan7.Pipeline(alphabet, background=background)
with pyhmmer.easel.SequenceFile("PDB FASTA Files/rcsb_pdb_1DBG.fasta", digital=True, alphabet=alphabet) as seq_file:
    hits = pipeline.search_hmm(hmm, seq_file)


ali = hits[0].domains[0].alignment
print(ali)



##########################################################
## secondary structure prediction scores
"""
g: light grey = coil
y: yellow = strand
p: pink = helix
u: no color = undefined
"""

# Known to have a right-handed beta-helix
pdb_1DBG_ps = import_txt("PSIPRED Annotation/pdb_1DBG_PSIPRED Annotation.txt")
pdb_1DAB_ps = import_txt("PSIPRED Annotation/pdb_1DAB_PSIPRED Annotation.txt")
pdb_1EA0_ps = import_txt("PSIPRED Annotation/pdb_1EA0_PSIPRED Annotation.txt")
pdb_1QJV_ps = import_txt("PSIPRED Annotation/pdb_1QJV_PSIPRED Annotation.txt")
pdb_1TYU_ps = import_txt("PSIPRED Annotation/pdb_1TYU_PSIPRED Annotation.txt")


# Predicted to have a beta-helix by UNIPROT
Q6LZ14_ps = import_txt('PSIPRED Annotation/Q6LZ14_PSIPRED Annotation.txt')
P35338_ps = import_txt('PSIPRED Annotation/P35338_PSIPRED Annotation.txt')
Q6ZGA1_ps = import_txt('PSIPRED Annotation/Q6ZGA1_PSIPRED Annotation.txt')


len(P35338)
len(Q6ZGA1)


MAKE A FUNCTION THAT CHECKS THE CODING P, Y, etc.


##########################################################
## segment length





##########################################################
## siklearn





