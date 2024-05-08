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
from autocorrect import Speller
spell = Speller(lang='en')


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
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from itertools import chain



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


# Known to have the beta-barrel fold: jellyroll
pdb_7VJV = functions.import_seq("PDB FASTA Files/rcsb_pdb_7VJV.fasta")
pdb_4C8D = functions.import_seq("PDB FASTA Files/rcsb_pdb_4C8D.fasta")


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


# Known to have the beta-barrel fold: jellyroll
pdb_7VJV_ps = functions.import_txt("PSIPRED Annotation/pdb_7VJV_PSIPRED Annotation.txt")
pdb_4C8D_ps = functions.import_txt("PSIPRED Annotation/pdb_4C8D_PSIPRED Annotation.txt")


# Predicted to have a beta-helix by UNIPROT
Q6LZ14_ps = functions.import_txt('PSIPRED Annotation/Q6LZ14_PSIPRED Annotation.txt')
P35338_ps = functions.import_txt('PSIPRED Annotation/P35338_PSIPRED Annotation.txt')
Q6ZGA1_ps = functions.import_txt('PSIPRED Annotation/Q6ZGA1_PSIPRED Annotation.txt')



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

Pattern: SI-B23-T3-B1-T1  -  B23-T3-B1-T1  -  ...  -  T3-B1-T1-B23-SI

s-B23 and s-B1 lengths are fixed as 9 and 3 respectively for two reasons:
    1. these are the numbers of residues shared by all known β-helices
    2. it helps limit the search space and reduce the computational costs
    
NOTE: the paper says s-B23 is 8 aa long, but the comparative is 9 aa.
    
s-T3 and s-T1 connect s-B23 and s-B1 and range from 1 to 80 amino acids. 
NOTE: β-helix structures will break if the insertion is too long

State s-I is the non-β-helix state at the start and end of the folding pattern.
This is arbirarily set to 5 aa as a starting frame to account for a head
and tail with s-I features in the parsing.

A protein w/o any β-helix structures is defined to be one single s-I node.


This code block will segment the normal amino acid sequence and the PSIPRED
score sequence.
"""

# the spacing for s-T1 and s-T3 regions differs from one another and
# vary from 1 to 80 aa. The following will randomize selection of
# segmentation commands for these folding regions.
T_spacing = list(chain( *[[1]*20, [5]*20, [10]*20, [15]*20, [20]*15, [25]*3, \
                          [30]*3, [35]*3, [40, 45, 50, 55, 60, 65, 70, 75, 80]] ))
B23_spacing = list(chain( *[[6]*3, [7]*3, [8]*15, [9]*3, [10] ] ))

iterations = 15



segmenting = {'s-B23': np.random.choice(T_spacing, iterations, replace=True), \
              's-T3': np.random.choice(T_spacing, iterations, replace=True), \
              's-B1': 3, 's-T1': np.random.choice(T_spacing, iterations, replace=True), \
              'start s-I': np.random.choice(T_spacing, iterations, replace=True), \
              'end s-I': np.random.choice(T_spacing, iterations, replace=True)}


# example sequence parsing into the amino acid form and PSIPRED scores
parsed, parsed_ps = functions.parse_protein(pdb_1DBG, pdb_1DBG_ps, iterations, segmenting)
    



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

Recall the pattern: SI-B23-T3-B1-T1  -  B23-T3-B1-T1  -  ...  -  T3-B1-T1-B23-SI
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


# some external actions are copied here from the example code
# included above
T_spacing = list(chain( *[[1]*20, [5]*20, [10]*20, [15]*20, [20]*15, [25]*3, \
                          [30]*3, [35]*3, [40, 45, 50, 55, 60, 65, 70, 75, 80]] ))
B23_spacing = list(chain( *[[6]*3, [7]*3, [8]*15, [9]*3 ], [10] ))

iterations = 15



segmenting = {'s-B23': np.random.choice(T_spacing, iterations, replace=True), \
              's-T3': np.random.choice(T_spacing, iterations, replace=True), \
              's-B1': 3, 's-T1': np.random.choice(T_spacing, iterations, replace=True), \
              'start s-I': np.random.choice(T_spacing, iterations, replace=True), \
              'end s-I': np.random.choice(T_spacing, iterations, replace=True)}
    


reg_exp_template = {'hydrophobic': ['A', 'F', 'I', 'L', 'M', 'V', 'W', 'Y'], 
                    'ionisable': ['D', 'E', 'R', 'K'],
                    'other': {}}



def annotate(protein, psipred_protein, reg_temp, iterations, T_segmenting, \
             alread_parsed, alread_parsed_ps, do_parse = True):
    
    if do_parse == True:
        parsed, parsed_ps = functions.parse_protein(protein, psipred_protein, \
                                                    iterations, T_segmenting)
        
        reg_protein = functions.regex_all_seg(parsed, reg_temp)
    
    else:
        parsed = alread_parsed
        parsed_ps = alread_parsed_ps
        
        reg_protein = functions.regex_all_seg(parsed, reg_temp)
        
        #reg_protein = [functions.regex_aa(reg_temp, x) for x in parsed]
        
    
    B2_T2_B3 = 'HXHXXIXHX'
    B1 = 'HXH'
    
    # Feature nodes
    # a. Regular expression template
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
            'B23_distance': distance_B23[0][i],
            'B23_mu': distance_B23[1],
            'B23_sigma': distance_B23[2]
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
        'B23_distance': None,
        'B23_mu': None,
        'B23_sigma': None
    }
    
    features.append(no_seg)


    return features



# example features calculation
final_features = annotate(pdb_1DBG, pdb_1DBG_ps, reg_exp_template, iterations, segmenting, None, None)



"""
The following proteins are entered into the protein database as comfirmed 
beta-helix folding proteins: 
    pdb_1DBG, pdb_1DAB, pdb_1EA0, pdb_1QJV, pdb_1TYU.
    
These will be labeled y = "beta-helix". pdb_1DBG and pdb_1EA0 will be used
to train the CRF.   

The following are proteins entered into the protein database as confirmed
beta-barrel (jellyroll) folding proteins:
    pdb_7VJV, pdb_4C8D
    
These will be labeled y = "beta-barrel" and bothwill be used to train the CRF.


The following are proteins entered into UNIPROT without experimental data
confirming their protein folding. These will be recycled with alternative
y-values outcomes, one for y = "beta-helix" and the other with y = "beta-barrel".
This will be used to test how well the CRF identifies them as one or the
other.
    Q6LZ14, P35338, Q6ZGA1
"""


# Known to have a right-handed beta-helix
pdb_1DBG_annotated = annotate(pdb_1DBG, pdb_1DBG_ps, reg_exp_template, iterations, segmenting, None, None)
pdb_1DAB_annotated = annotate(pdb_1DAB, pdb_1DAB_ps, reg_exp_template, iterations, segmenting, None, None)
pdb_1EA0_annotated = annotate(pdb_1EA0, pdb_1EA0_ps, reg_exp_template, iterations, segmenting, None, None)
pdb_1QJV_annotated = annotate(pdb_1QJV, pdb_1QJV_ps, reg_exp_template, iterations, segmenting, None, None)
pdb_1TYU_annotated = annotate(pdb_1TYU, pdb_1TYU_ps, reg_exp_template, iterations, segmenting, None, None)


# Known to have the beta-barrel fold: jellyroll
pdb_7VJV_annotated = annotate(pdb_7VJV, pdb_7VJV_ps, reg_exp_template, iterations, segmenting, None, None)
pdb_4C8D_annotated = annotate(pdb_4C8D, pdb_4C8D_ps, reg_exp_template, iterations, segmenting, None, None)


# Predicted to have a beta-helix by UNIPROT
Q6LZ14_annotated = annotate(Q6LZ14, Q6LZ14_ps, reg_exp_template, iterations, segmenting, None, None)
P35338_annotated = annotate(P35338, P35338_ps, reg_exp_template, iterations, segmenting, None, None)
Q6ZGA1_annotated = annotate(Q6ZGA1, Q6ZGA1_ps, reg_exp_template, iterations, segmenting, None, None)




##########################################################
## PDB Segmentation
"""
Using the structure image rendition of 1DBG and 1QJV in the PDB
the following "actual" segmentations were rendered for model training.

1DBG: https://www.rcsb.org/3d-view/1DBG
1QJV: https://www.rcsb.org/3d-view/1QJV
"""

actual_1DBG = [[pdb_1DBG[:45], 
               pdb_1DBG[45:55],   pdb_1DBG[55:57],   pdb_1DBG[57:61],   pdb_1DBG[61:70], 
               pdb_1DBG[70:85],   pdb_1DBG[85:87],   pdb_1DBG[87:91],   pdb_1DBG[91:95],
               pdb_1DBG[95:106],  pdb_1DBG[106:120], pdb_1DBG[120:123], pdb_1DBG[123:128], 
               pdb_1DBG[128:136], pdb_1DBG[136:145], pdb_1DBG[145:148], pdb_1DBG[148:160],
               pdb_1DBG[160:168], pdb_1DBG[168:176], pdb_1DBG[176:179], pdb_1DBG[179:198], 
               pdb_1DBG[198:203], pdb_1DBG[203:207], pdb_1DBG[207:217], pdb_1DBG[217:220],
               pdb_1DBG[220:235], pdb_1DBG[235:240], pdb_1DBG[240:244], pdb_1DBG[244:250], 
               pdb_1DBG[250:259], pdb_1DBG[259:262], pdb_1DBG[262:266], pdb_1DBG[266:272],
               pdb_1DBG[272:281], pdb_1DBG[281:285], pdb_1DBG[285:294], pdb_1DBG[294:297], 
               pdb_1DBG[297:307], pdb_1DBG[307:311], pdb_1DBG[311:320], pdb_1DBG[320:325],
               pdb_1DBG[325:345], pdb_1DBG[345:348], pdb_1DBG[348:352], pdb_1DBG[352:356], 
               pdb_1DBG[356:386], pdb_1DBG[386:389], pdb_1DBG[389:398], pdb_1DBG[398:400], 
               pdb_1DBG[400:] ]]
    
# 400 - 414 aa seemed to form another s-B23 segment, but was not paired with a
# full sementation profile, and so it was absorbed with the last s-I region.
    
actual_1DBG_ps = [[pdb_1DBG_ps[:45], 
                 pdb_1DBG_ps[45:55],   pdb_1DBG_ps[55:57],   pdb_1DBG_ps[57:61],   pdb_1DBG_ps[61:70], 
                 pdb_1DBG_ps[70:85],   pdb_1DBG_ps[85:87],   pdb_1DBG_ps[87:91],   pdb_1DBG_ps[91:95],
                 pdb_1DBG_ps[95:106],  pdb_1DBG_ps[106:120], pdb_1DBG_ps[120:123], pdb_1DBG_ps[123:128], 
                 pdb_1DBG_ps[128:136], pdb_1DBG_ps[136:145], pdb_1DBG_ps[145:148], pdb_1DBG_ps[148:160],
                 pdb_1DBG_ps[160:168], pdb_1DBG_ps[168:176], pdb_1DBG_ps[176:179], pdb_1DBG_ps[179:198], 
                 pdb_1DBG_ps[198:203], pdb_1DBG_ps[203:207], pdb_1DBG_ps[207:217], pdb_1DBG_ps[217:220],
                 pdb_1DBG_ps[220:235], pdb_1DBG_ps[235:240], pdb_1DBG_ps[240:244], pdb_1DBG_ps[244:250], 
                 pdb_1DBG_ps[250:259], pdb_1DBG_ps[259:262], pdb_1DBG_ps[262:266], pdb_1DBG_ps[266:272],
                 pdb_1DBG_ps[272:281], pdb_1DBG_ps[281:285], pdb_1DBG_ps[285:294], pdb_1DBG_ps[294:297], 
                 pdb_1DBG_ps[297:307], pdb_1DBG_ps[307:311], pdb_1DBG_ps[311:320], pdb_1DBG_ps[320:325],
                 pdb_1DBG_ps[325:345], pdb_1DBG_ps[345:348], pdb_1DBG_ps[348:352], pdb_1DBG_ps[352:356], 
                 pdb_1DBG_ps[356:386], pdb_1DBG_ps[386:389], pdb_1DBG_ps[389:398], pdb_1DBG_ps[398:400], 
                 pdb_1DBG_ps[400:] ]]
    
    

real_pdb_1DBG_annotated = annotate(pdb_1DBG, pdb_1DBG_ps, reg_exp_template, \
                              iterations, segmenting, actual_1DBG, actual_1DBG_ps, \
                              do_parse = False)
# Notice that because this entry is one dimentional (only one, true
# segmentation is given), then B23_distance represents the simple mean.
# We adjust this as follows.

segmentation_mean = pdb_1DBG_annotated[0]['B23_mu']
segmentation_sigma = pdb_1DBG_annotated[0]['B23_sigma']

real_pdb_1DBG_annotated[0]['B23_distance'] = \
        round((real_pdb_1DBG_annotated[0]['B23_distance'] - \
               segmentation_mean) / segmentation_sigma, 3)





actual_1QJV = [[pdb_1QJV[:34], 
               pdb_1QJV[34:45],   pdb_1QJV[45:47],   pdb_1QJV[47:50],   pdb_1QJV[50:54],   
               pdb_1QJV[54:69],   pdb_1QJV[69:90],   pdb_1QJV[90:93],   pdb_1QJV[93:98],
               pdb_1QJV[98:107],  pdb_1QJV[107:132], pdb_1QJV[132:135], pdb_1QJV[135:141], 
               pdb_1QJV[141:150], pdb_1QJV[150:155], pdb_1QJV[155:158], pdb_1QJV[158:161], 
               pdb_1QJV[161:172], pdb_1QJV[172:176], pdb_1QJV[176:179], pdb_1QJV[179:181], 
               pdb_1QJV[181:192], pdb_1QJV[192:207], pdb_1QJV[207:210], pdb_1QJV[210:219], 
               pdb_1QJV[219:229], 
               pdb_1QJV[229:] ]]

# 229 - 268 aa seemed to form another s-B23 and s-T2 segments, but was not 
# paired with a full sementation profile, and so it was absorbed with the 
# last s-I region.
    
actual_1QJV_ps = [[pdb_1QJV_ps[:34], 
                  pdb_1QJV_ps[34:45],   pdb_1QJV_ps[45:47],   pdb_1QJV_ps[47:50],   pdb_1QJV_ps[50:54],   
                  pdb_1QJV_ps[54:69],   pdb_1QJV_ps[69:90],   pdb_1QJV_ps[90:93],   pdb_1QJV_ps[93:98],
                  pdb_1QJV_ps[98:107],  pdb_1QJV_ps[107:132], pdb_1QJV_ps[132:135], pdb_1QJV_ps[135:141], 
                  pdb_1QJV_ps[141:150], pdb_1QJV_ps[150:155], pdb_1QJV_ps[155:158], pdb_1QJV_ps[158:161], 
                  pdb_1QJV_ps[161:172], pdb_1QJV_ps[172:176], pdb_1QJV_ps[176:179], pdb_1QJV_ps[179:181], 
                  pdb_1QJV_ps[181:192], pdb_1QJV_ps[192:207], pdb_1QJV_ps[207:210], pdb_1QJV_ps[210:219], 
                  pdb_1QJV_ps[219:229], 
                  pdb_1QJV_ps[229:] ]]


real_pdb_1QJV_annotated = annotate(pdb_1QJV, pdb_1QJV_ps, reg_exp_template, \
                              iterations, segmenting, actual_1QJV, actual_1QJV_ps, \
                              do_parse = False)
# Notice that because this entry is one dimentional (only one, true
# segmentation is given), then B23_distance represents the simple mean.
# We adjust this as follows.

segmentation_mean = pdb_1QJV_annotated[0]['B23_mu']
segmentation_sigma = pdb_1QJV_annotated[0]['B23_sigma']

real_pdb_1QJV_annotated[0]['B23_distance'] = \
        round((real_pdb_1QJV_annotated[0]['B23_distance'] - \
               segmentation_mean) / segmentation_sigma, 3)



##########################################################
## CRF funtion settings

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)



##########################################################
## CRF for all compiled


labels = ['s-B23', 's-B23', 's-B1', 's-B1', 'whole', 'whole', 'whole', \
          'fold only', 'fold only', 'fold only', 's-B23', 's-B23', 's-B23', \
          's-T3', 's-T3', 's-T3', 's-B1', 's-B1', 's-B1', 's-T1', 's-T1', \
          's-T1', 's-T1', 's-T3', 's-B23', 's-B23', 's-B23']

    

X_train = [real_pdb_1DBG_annotated[0]] + [real_pdb_1QJV_annotated[0]]

y_train = [labels] * 2



X_test_joined = pdb_1DBG_annotated[0:iterations] + pdb_1DAB_annotated[0:iterations] + \
                pdb_1EA0_annotated[0:iterations] + pdb_1QJV_annotated[0:iterations] + \
                pdb_1TYU_annotated[0:iterations] + Q6LZ14_annotated[0:iterations] + \
                P35338_annotated[0:iterations] + Q6ZGA1_annotated[0:iterations]

y_test_joined = ([labels] * iterations) * 8




crf.fit(X_train, y_train)
y_pred = crf.predict(X_test_joined)


metrics.flat_f1_score(y_test_joined, y_pred, average='weighted')

by_protein, by_segment = functions.crf_match(y_test_joined, y_pred, 8, 15)


# match = []
# for i in range(len(y_pred)):
#     match.append(all([x == 'beta-helix' for x in y_pred[i]]))

# all(match)


##########################################################
## CRF for all compiled - simple labels

labels_simple = ['beta-helix'] * len(labels)


y_train_simple = [labels_simple] * 2
y_test_joined_simple = ([labels_simple] * iterations) * 8



crf.fit(X_train, y_train_simple)
y_pred = crf.predict(X_test_joined)


metrics.flat_f1_score(y_test_joined_simple, y_pred, average='weighted')

by_protein, by_segment = functions.crf_match(y_test_joined_simple, y_pred, 8, 15)







