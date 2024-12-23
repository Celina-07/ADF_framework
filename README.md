# ADF_framework

The different steps are the following:

![ADF_workflow](workflow.pdf)


Here is the organization of the code:
* **data**: contains all necessary data files for both Adressa and MIND datasets. 
* **reco_scores_baseline**: contains the list of relevance scores for all recommendation models (centroidVector, LSTUR, NAML, NPA, NRMS). Please note that news centroidVector was applied using the [ClayRS library](https://github.com/swapUniba/ClayRS), while deep-learning models were applied using the [NewsRecLib library](https://github.com/andreeaiana/newsreclib/tree/main). 
* **re_ranking**: contains the notebooks to re-rank the recommendations. Different ways to re-rank recommendations are included:
  * *Greeedy re-ranking*: serve as baseline.
  * *ADF*: proposed approach, fairness-constrained diversification, with personalization of the target diversity.
* **reco_scores_ADF**: contains the re-ranking list of recommendations when ADF is applied.
* **reco_scores_greedy**: contains the re-ranking list of recommendations when the greedy diversification is applied.
* **evaluation**: contains a notebook necessary to evaluate the outputs of the models (baselines and ADF).


# License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

[![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-sa/4.0/)
