# -*- coding: utf-8 -*-
"""
For the EN.625.692.81.SP24 Probabilistic Models Semester Project
By Shelby Golden April 27th, 2024

This is the primary analysis script used for the paper. The setup is based on
a 'topic paper', which is 'Protein Fold Recognition Using Segmentation 
Conditional Random Fields (SCRFs)' by Liu et al 2006.

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
import Custom_Functions as functions
import numpy as np
import random
import re
#import pyhmmer



##########################################################
##########################################################
## import sequences and segmentation

##########################################################
## protein primary sequence

# Known to have a right-handed beta-helix
pdb_1DBG = functions.import_seq("PDB FASTA Files/rcsb_pdb_1DBG.fasta")
pdb_1DAB = functions.import_seq("PDB FASTA Files/rcsb_pdb_1DAB.fasta")
pdb_1EA0 = functions.import_seq("PDB FASTA Files/rcsb_pdb_1EA0.fasta")
pdb_1QJV = functions.import_seq("PDB FASTA Files/rcsb_pdb_1QJV.fasta")
pdb_1TYU = functions.import_seq("PDB FASTA Files/rcsb_pdb_1TYU.fasta")


# Predicted to have a beta-helix by UNIPROT
Q6LZ14 = functions.import_seq("PDB FASTA Files/UNIPROT_Q6LZ14.fasta")
P35338 = functions.import_seq("PDB FASTA Files/UNIPROT_P35338.fasta")
Q6ZGA1 = functions.import_seq("PDB FASTA Files/UNIPROT_Q6ZGA1.fasta")



##########################################################
## PSIPRED scores
"""
g: light grey = coil
y: yellow = sheet (strand)
p: pink = helix
u: no color = undefined
"""

# Known to have a right-handed beta-helix
pdb_1DBG_ps = functions.import_txt("PSIPRED Annotation/pdb_1DBG_PSIPRED Annotation.txt")
pdb_1DAB_ps = functions.import_txt("PSIPRED Annotation/pdb_1DAB_PSIPRED Annotation.txt")
pdb_1EA0_ps = functions.import_txt("PSIPRED Annotation/pdb_1EA0_PSIPRED Annotation.txt")
pdb_1QJV_ps = functions.import_txt("PSIPRED Annotation/pdb_1QJV_PSIPRED Annotation.txt")
pdb_1TYU_ps = functions.import_txt("PSIPRED Annotation/pdb_1TYU_PSIPRED Annotation.txt")


# Predicted to have a beta-helix by UNIPROT
Q6LZ14_ps = functions.import_txt('PSIPRED Annotation/Q6LZ14_PSIPRED Annotation.txt')
P35338_ps = functions.import_txt('PSIPRED Annotation/P35338_PSIPRED Annotation.txt')



# The following is used to confirm that the input entries for the PSIPRED
# scores only contain the expected letters. Two criteria are used, with the
# first representing the ideal if there are no undefined amino acid results.

# most PSIPRED results contined only the target g, y, and p results
allowed_1 = set('g' + 'y' + 'p')

# some continaed additional residues called u
allowed_2 = set('g' + 'y' + 'p' + 'u')

functions.check(pdb_1EA0_ps, allowed_2)



##########################################################
## Segmentation
"""
Section 4.1. Protein structural graph for β-helix in the topic paper 
has notations about restrictions for the length for some of the
components for the fold segments. Arbitrarily, s-I is set to around 10
to allow for some preceeding and ending s-I tails on the sequence folding.

Patern: SI-B23-T3-B1-T1  -  B23-T3-B1-T1  -  ...  -  T3-B1-T1-B23-SI

s-B23 and s-B1 lengths are fixed as 9 and 3 respectively for two reasons:
    1. these are the numbers of residues shared by all known β-helices
    2. it helps limit the search space and reduce the computational costs
    
NOTE: the paper says s-B23 is 8 bp long, but the comparative is 9 bp.
    
s-T3 and s-T1 connect s-B23 and s-B1 and range from 1 to 80 amino acids. 
NOTE: β-helix structures will break if the insertion is too long

State s-I is the non-β-helix state at the start and end of the folding patern.
This is arbirarily set to 5 bp as a starting frame to account for a head
and tail with s-I features in the parsing.

A protein w/o any β-helix structures is defined to be one single s-I node.


This code block will segment the normal amino acid sequence and the PSIPRED
score sequence.
"""

# the spacing for s-T1 and s-T3 regions differs from one another and
# vary from 1 to 80 bp. The following will randomize selection of
# segmentation commands for these folding regions.
spacing = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]


segmenting = {'s-B23': 8, 's-T3': random.sample(spacing, len(spacing)), 
              's-B1': 3, 's-T1': random.sample(spacing, len(spacing)), 's-I': 5}


# example sequence parsing into the amino acid form and PSIPRED scores
parsed, parsed_ps = functions.parse_protein(pdb_1DBG, pdb_1DBG_ps, spacing, segmenting)
    


##########################################################
##########################################################
## Feature extraction - Nodes

##########################################################
## a. Regular expression template
"""
The topic paper is vague about what is exactly calculated here. In this
application, matches are found for all segments and the counts "inside" and
"outside" of the expected segmenting region is calcuated.

Note: the s-B1 regex feature is not provided in the paper. Based on one protein
known to have the beta-helix folding (1DAB) it is estimated to have the
regex = 'HXH'
source: https://www.rcsb.org/3d-view/1DAB
"""

B2_T2_B3 = 'HXHXXIXHX'
B1 = 'HXH'

reg_exp_template = {'hydrophobic': ['A', 'F', 'I', 'L', 'M', 'V', 'W', 'Y'], 
                    'ionisable': ['D', 'E', 'R', 'K'],
                    'other': {}}


# example regular expression conversion
reg_parsed = functions.regex_all_seg(parsed, reg_exp_template)

# example alignment scores
inside, outside = functions.check_alignment(reg_parsed, B2_T2_B3)
inside, outside = functions.check_alignment(reg_parsed, B1)



# for troubleshooting only
# function that shows all alignments detected
#aligner_1 = Align.PairwiseAligner(match_score=1.0)
#alignments = aligner_1.align(target, query)

#for alignment in alignments:
#     print(alignment)

#alignment = alignments[0]
#print(alignment)



##########################################################
## b. Probabilistic HMM profiles
"""
The pyHMMER package only allows for alphabet profiles that match cononical
amino acid, RNA, and DNA profiles as developed by the Easel C library:
http://eddylab.org/ . These are the same people who developed the  HMMER
webpage: https://www.ebi.ac.uk/Tools/hmmer/about .

Therefore, the regex format from above is adjusted to an arbitrary nucleotide
available in the pyhmmer.easel.Alphabet.dna() class.

Due to challenges implementing this feature expression, this section
was abandoned for the sake of time to complete the project.
"""

# B2_T2_B3_dna = 'CTCTTATCT'
# B1_dna = 'CTC'


# example = reg_parsed[1][2]

# example = example.replace("I", "A")
# example = example.replace("H", "C")
# example = example.replace("X", "T")


  
# seq1 = pyhmmer.easel.TextSequence(name=b"seq1", sequence="ACCGACA")
# seq2 = pyhmmer.easel.TextSequence(name=b"seq2", sequence="GGGCCAACA")

# rna = pyhmmer.easel.Alphabet.rna()
# dig1, dig2 = [s.digitize(rna) for s in [seq1, seq2]]
# builder = pyhmmer.plan7.Builder(rna, prior_scheme="alphabet")

# gen = pyhmmer.hmmer.([dig1], [dig2], builder=builder)
# next(gen)




# with pyhmmer.easel.SequenceFile("938293.PRJEB85.HG003687.faa", digital=True) as seq_file:
#     sequences = list(seq_file)

# with pyhmmer.plan7.HMMFile("KR.hmm") as hmm_file:
#     for hits in pyhmmer.hmmsearch(hmm_file, sequences, cpus=4):
#       print(f"HMM {hits.query_name.decode()} found {len(hits)} hits in the target sequences")





# alphabet = pyhmmer.easel.Alphabet.dna()

# seq1 = pyhmmer.easel.TextSequence(name=b'seq1', sequence=B2_T2_B3_dna)
# seq2 = pyhmmer.easel.TextSequence(name=b'seq2', sequence=B2_T2_B3_dna)

# msa  = pyhmmer.easel.TextMSA(name=b'msa', sequences=[seq1, seq2])


# msa_d = msa.digitize(alphabet)

# builder = pyhmmer.plan7.Builder(alphabet)
# background = pyhmmer.plan7.Background(alphabet)
# hmm, _, _ = builder.build_msa(msa_d, background)


# seq_seg = pyhmmer.easel.TextSequence(name=b"seq_seg", sequence=example)
# dig_seg = seq_seg.digitize(alphabet)


    
# seq = [dig_seg, dig_seg]

# pipeline = pyhmmer.plan7.Pipeline(alphabet, background=background)    
# hits = pyhmmer.hmmsearch(hmm, seq)
# hits = pipeline.search_hmm(hmm, seq)


# ali = hits[0].domains[0].alignment
# print(ali)



# pipeline = pyhmmer.plan7.Pipeline(alphabet, background=background)
# with pyhmmer.easel.SequenceFile("LuxC.faa", digital=True, alphabet=pyhmmer.easel.Alphabet.amino()) as seq_file:
#     hits = pipeline.search_hmm(hmm, seq_file)
    



# ##########################################################
## c. Secondary structure prediction scores
"""
PSIPRED results give an image of the structure most likely to be associated with
any of given residue in the amino acid sequence. These usually come to be
either a helix, sheet (strand), and coil shape. The coding for this is below.
Some residues had no apparent prediction, and these are labeled as undefined.

Because the prediction score is not given as a value, each segment will be
converted into a binary for each feature associated with == or != helix,
sheet, or coil. Then the average number of residues associated with that
outcome is saved as the PSIPRED score.

g: light grey = coil
y: yellow = sheet (strand)
p: pink = helix
u: no color = undefined

The output for this feature is a little complex. It is first divided by
the calculations done by each candidate sementation. For each instantiation
of segmentation, there are five probability outcomes with three values each.

    1. the whole region available for folding without the s-I regions
    2. all s-B23 regions together
    3. all s-T3 regions together
    4. all s-B1 regions together
    5. all s-T1 regions together

for each subset, the mean for helix, sheet, and coil are calculated, in that
exact order. Each value is calculated using the simple freq / total calculation.
"""

# example secondary prediction scores over segments
fold_prob = functions.secondary_scores(parsed_ps, pdb_1DBG_ps)



##########################################################
## d. Segment length
"""
The feature functions fL1(x,si) and fL3(x,si) are the estimated density of 
length (si ) under the distribution of length (s-T1) and length (s-T3) 
respectively. The distribution is the Asymmetric Exponential, or the 
Asymmetric Laplace distribution. The parameters are not given, and so they are
guessed, as shown below. Then those density paramters are used to provide
that feature outcome.

Recall the patern: SI-B23-T3-B1-T1  -  B23-T3-B1-T1  -  ...  -  T3-B1-T1-B23-SI
"""
from scipy.stats import laplace_asymmetric
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# Plot of guessed parameters
fig, ax = plt.subplots(1, 1)

x = np.linspace(laplace_asymmetric.ppf(0.01, kappa=0.4, loc=4, scale=0.6),
                laplace_asymmetric.ppf(0.99, kappa=0.4, loc=4, scale=0.6), 100)

ax.plot(x, laplace_asymmetric.pdf(x, kappa=0.4, loc=5, scale=1.6),
       'r-', lw=2, label='k=0.4, loc=5, scale=1.6')

ax.plot(x, laplace_asymmetric.pdf(x, kappa=0.5, loc=5, scale=0.6),
       'b-', lw=2, label='k=0.5, loc=5, scale=0.6')

ax.plot(x, laplace_asymmetric.pdf(x, kappa=0.6, loc=5, scale=1),
       'k-', lw=2, label='k=0.6, loc=5, scale=1')


ax.set_xlim([x[0], x[-1]])
ax.set_ylim([0, 1])

ax.set_xlabel("0.01 to 0.99 PPF")
ax.set_ylabel("Density")
ax.set_title("Density Outcomes for Candidate Parameters \n Asymmetric Laplace(kappa, location, scale)")

ax.legend(loc='best', frameon=False)


plt.show()


# Best candidate for s-T1 is the blue line
mean_T1, var_T1, skew_T1, kurt_T1 = laplace_asymmetric.stats(kappa=0.5, loc=5, scale=0.6, moments='mvsk')

# Best candidate for s-T3 is the black line
mean_T3, var_T3, skew_T3, kurt_T3 = laplace_asymmetric.stats(kappa=0.6, loc=5, scale=1, moments='mvsk')



# example segment length
T1_score, T3_score = functions.laplace_prob(parsed)



##########################################################
##########################################################
## Feature extraction - Internodes

##########################################################
## c. Distance between adjacent s-B23 segments

# example distance
functions.dist_B23(parsed)



##########################################################
##########################################################
## CRF

##########################################################
## Compile the features for the CRF wrapper


# the spacing for s-T1 and s-T3 regions differs from one another and
# vary from 1 to 80 bp. The following will randomize selection of
# segmentation commands for these folding regions.
spacing = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]


segmenting = {'s-B23': 8, 's-T3': random.sample(spacing, len(spacing)), 
              's-B1': 3, 's-T1': random.sample(spacing, len(spacing)), 's-I': 5}


reg_exp_template = {'hydrophobic': ['A', 'F', 'I', 'L', 'M', 'V', 'W', 'Y'], 
                    'ionisable': ['D', 'E', 'R', 'K'],
                    'other': {}}



def annotate(protein, psipred_protein, reg_temp, T_spacing, T_segmenting):
    
    parsed, parsed_ps = functions.parse_protein(protein, psipred_protein, T_spacing, T_segmenting)
    B2_T2_B3 = 'HXHXXIXHX'
    B1 = 'HXH'
    
    # Feature nodes
    # a. Regular expression template
    reg_protein = functions.regex_all_seg(parsed, reg_temp)
    
    
    inside_B23, outside_B23 = functions.check_alignment(reg_protein, B2_T2_B3)
    inside_B1, outside_B1 = functions.check_alignment(reg_protein, B1)
    
    # c. Secondary structure prediction scores
    fold_prob = functions.secondary_scores(parsed_ps, psipred_protein)
    
    # d. Segment length
    T1_score, T3_score = functions.laplace_prob(parsed)
    
    
    # Feature internodes    
    distance_B23 = functions.dist_B23(parsed)
    
    
    features = []
    for i in range(len(parsed)):
        seg_features = {
            # Feature nodes
            # a. Regular expression template
            'in_B23': outside_B23[i],
            'out_B23': outside_B23[i],
            'in_B1': inside_B1[i],
            'out_B1': outside_B1[i],
            
            # c. Secondary structure prediction scores
            'whole_helix': fold_prob[len(parsed)][0][0],
            'whole_sheet': fold_prob[len(parsed)][0][1],
            'whole_coil': fold_prob[len(parsed)][0][2],
            
            'fold_helix': fold_prob[i][1][0],
            'fold_sheet': fold_prob[i][1][1],
            'fold_coil': fold_prob[i][1][2],
            
            'B23_helix': fold_prob[i][2][0],
            'B23_sheet': fold_prob[i][2][1],
            'B23_coil': fold_prob[i][2][2],
            
            'T3_helix': fold_prob[i][3][0],
            'T3_sheet': fold_prob[i][3][1],
            'T3_coil': fold_prob[i][3][2],
            
            'B1_helix': fold_prob[i][4][0],
            'B1_sheet': fold_prob[i][4][1],
            'B1_coil': fold_prob[i][4][2],
            
            'T1_helix': fold_prob[i][5][0],
            'T1_sheet': fold_prob[i][5][1],
            'T1_coil': fold_prob[i][5][2],
            
            # d. Segment length
            'T1_score': T1_score[i],
            'T3_score': T3_score[i],
            
            # Feature internodes    
            'B23_distance': distance_B23
        }
        
        features.append(seg_features)
        
        
        
    whole_reg_protein = functions.regex_aa(reg_temp, protein)
    
    # No segmentation, the whole sequence is termed s-I
    no_seg = {
        # Feature nodes
        # a. Regular expression template
        'in_B23': None,
        'out_B23': len( re.findall(B2_T2_B3, whole_reg_protein) ),
        'in_B1': None,
        'out_B1': len( re.findall(B1, whole_reg_protein) ),
        
        # c. Secondary structure prediction scores
        'whole_helix': fold_prob[len(parsed)][0][0],
        'whole_sheet': fold_prob[len(parsed)][0][1],
        'whole_coil': fold_prob[len(parsed)][0][2],
        
        'fold_helix': fold_prob[len(parsed)][1][0],
        'fold_sheet': fold_prob[len(parsed)][1][1],
        'fold_coil': fold_prob[len(parsed)][1][2],
        
        'B23_helix': fold_prob[len(parsed)][2][0],
        'B23_sheet': fold_prob[len(parsed)][2][1],
        'B23_coil': fold_prob[len(parsed)][2][2],
        
        'T3_helix': fold_prob[len(parsed)][3][0],
        'T3_sheet': fold_prob[len(parsed)][3][1],
        'T3_coil': fold_prob[len(parsed)][3][2],
        
        'B1_helix': fold_prob[len(parsed)][4][0],
        'B1_sheet': fold_prob[len(parsed)][4][1],
        'B1_coil': fold_prob[len(parsed)][4][2],
        
        'T1_helix': fold_prob[len(parsed)][5][0],
        'T1_sheet': fold_prob[len(parsed)][5][1],
        'T1_coil': fold_prob[len(parsed)][5][2],
        
        # d. Segment length
        'T1_score': None,
        'T3_score': None,
        
        # Feature internodes    
        'B23_distance': None
    }
    
    features.append(no_seg)


    return features




annotate(pdb_1DBG, pdb_1DBG_ps, reg_exp_template, spacing, segmenting)





