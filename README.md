# kNNDissimilarities
Repository containing the code for paper: Improving novelty and diversity of nearest-neighbors recommendation by exploiting dissimilarities for ECIR 2025. IR4Good Track.


## Getting Started
Prerequisites:
- Java 8 (at least)
- Mono (for Mymedialite)


## Instructions
 - git clone https://github.com/pablosanchezp/kNNDissimilarities.git
 - cd kNNDissimilarities
 - Run the following scripts in the terminal:
    - step1_download.sh
    - step2_splits.sh
    - step3_generate_recommenders_AmazonVynils.sh
    - step3_generate_recommenders_GoodReads.sh
    - step3_generate_recommenders_Lastfm.sh
    - step3_generate_recommenders_Movielens20M.sh


## Additional algorithms/frameworks used in the experiments
  * [MyMedialite](http://www.mymedialite.net/) (for BPR)
  * [Embarrassingly Shallow Autoencoders for Sparse Data](https://dl.acm.org/doi/10.1145/3308558.3313710) (for EASEr) [Source code](https://github.com/Darel13712/ease_rec?tab=readme-ov-file). Directory ease_rec
  * [Updatable, Accurate, Diverse, and Scalable Recommendations for Interactive Applications](https://dl.acm.org/doi/10.1145/2955101) (for RP3beta) [Source code](https://github.com/StivenMetaj/Recommender_Systems_Challenge_2020). Directory recommender-systems



## Authors
- Pablo Sánchez - [Universidad Pontificia Comillas](https://www.comillas.edu/)
- Javier Sanz-Cruzado - [University of Glasgow](https://www.gla.ac.uk/)
- Alejandro Bellogín - [Universidad Autónoma de Madrid](https://uam.es)


## Contact

* **Pablo Sánchez** - <psperez@icai.comillas.edu>


## Acknowledgments
 - This work was supported by grant PID2022-139131NB-I00 funded by MCIN/AEI/10.13039/501100011033 and by ``ERDF A way of making Europe.''
