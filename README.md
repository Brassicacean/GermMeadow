This was a simulation I made to help answer the question, "how can unconditional altruism evolve?" I imagined simple organisms called "germs", which reproduce asexually and have Mendelian inheritance. During each generation, they experience a selection event which kills some of the germs with a random probability. Before this selection event, germs with the A allele can "help" a neighbor by giving a boost to its survival probability at the expense of their own survival odds. Surviving germs reproduce with a random neighbor, giving the offspring a random allele from each parent.
Typically, the populations tend to become fractured early on, as regions with low density go extinct. These fragmented communities then reproduce in isolation and undergo trait fixation, meaning one of the two alleles goes extinct in each population. Then the selective forces determining the relative success of the alleles are acting on the community level instead of the individual.
To avoid making unnecessary comparisons between germs that are way too far to interact, I partitioned the field into a grid of partitions with length equal to the maximum interaction distance and limited the number of pair-wise comparisons to only germs within adjacent partitions. 

Dependencies: matplotlib, numpy
Innstructions:
Run GermSim6.py on the command line and select the number of AA (homozygous altruist), Aa (herterozygous), and aa (homozygous non-altruist) inndividual upon the prompt. To let a generation pass, you must exit the matplotlib popup window that shows that germs. 

Legend:
blue: AA
purple: Aa
red: aa
