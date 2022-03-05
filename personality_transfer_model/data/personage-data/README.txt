
PERSONAGE Dataset - Readme
--------------------------

Francois Mairesse & Marilyn A. Walker, 
University of Sheffield, 2006-2008


The PERSONAGE dataset contains a total of 580 utterances annotated
with personality ratings from human judges. The ratings were obtained
by using the Ten-Item Personality Inventory (Gosling et al., 2003) to
assess the personality of an hypothetical speaker for each utterance
(shown in a written form), providing ratings are on a scale from 1
(low) to 7 (high) for each of the Big Five personality trait (260
utterances only have extraversion ratings). The utterances were
generated using the PERSONAGE generator, the data thus also includes a
listing of the generation decisions being made to produce the
utterance, as well as the intermediary content plan tree, sentence
plan tree and final syntactic structure for each utterance. This data
was used for evaluating the style conveyed by the PERSONAGE-RB
rule-based generator, as well as for training the ranking models of
the PERSONAGE-OS generator and the parameter estimation models of the
PERSONAGE-PE data-driven generator. More information on the base
generator and the data-driven generation methods can be found in the
ACL papers (Mairesse & Walker, 2007 and 2008) as well as in Francois
Mairesse's PhD thesis, available at http://mi.eng.cam.ac.uk/~farm2.

The data is split into two xml files. The first
one--predefinedParams.xml--contains 240 utterances generated using
predefined parameters settings suggested by the psychology litterature
(with 15% gaussian noise) and 20 utterances generated using the SPaRKy
sentence planner (Stent et al., 2004). There are 80 utterances
manipulating extraversion, and 40 utterances manipulating each other
four trait. The second file--randomParams.xml--contains utterances
generated using uniformly distributed parameter values. The xml files
use the following structure:


<textplan>: Content/text plan (specific set of restaurants to
recommend or compare).

<alternative>: Individual utterances generated from the content
plan. The id of random utterance is prefixed with "random-", while the
utterances generated with parameters conveying each end of the Big
Five traits are prefixed with "$TRAIT_$END", e.g. "extra_low-" for
introversion parameters. The SPaRKy utterances are prefixed with
"sparky-".
			
<inputparameters>: PERSONAGE's target parameter values, on a scale
from 0 to 1. More details in the thesis.

<outputparameters>: PERSONAGE's true parameter values, i.e. the values
that were actually used at generation time. They might differ from
values above due to internal constraints.

<rstplan>: Rhetorical structure tree, with rhetorical relations
linking content propositions.

<sentenceplan>: Sentence plan tree, with aggregation operations
(e.g. BECAUSE) assigned to each rhetorical relation (e.g. JUSTIFY).

<dsslist>: List of resulting Deep Syntactic Structures (DSyntS's, one
per sentence), resulting from the aggregation, pragmatic marker
insertion and lexical choice components. The DSyntS format is detailed
in the RealPro user manual.

<realization>: Output realization of the list of DSyntS's, obtained
using RealPro.

<ratings>: The personality and naturalness ratings for each judge
(userA, userB, etc.), as well
as the ratings averaged over all judges (avg). Extra = extraversion, ems =
emotional stability, agree = agreeableness, consc = conscientiousness,
open = openness to experience. Early experiments only involved evaluating
extraversion, thus 260 utterances only contain ratings for that trait.


The utterances are also stored in a more compact tab-delimited format
(files predefinedParams.tab and randomParams.tab), with only the
average ratings for each trait and the naturalness, as well as the
utterance's realization.

Please contact the authors (farm2@cam.ac.uk) if you have any
question. We would also be interested in hearing about any experiment
done using this data.










