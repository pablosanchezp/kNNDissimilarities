#!/bin/bash


java_command=java
JAR=kNNDissimilarities.jar
jvm_memory=-Xmx200G



datasets="Movielens20M"

splits="RandomGlobal8020"


items_recommended=100
rec_strat="T_I" # T_I is train items
rec_prefix=rec


all_neighbours="20 40 60 80 100 120"



# HKV
k_factors="10 50 100"
lambda_factorizers="0.1 1 10"
alpha_factorizers="0.1 1 10"




mymedialite_path=MyMediaLite-3.11/bin
bpr_factors=$k_factors
bpr_bias_regs="0 0.5 1"
bpr_learn_rate=0.05
bpr_iter=50
bpr_regs_u="0.0025 0.001 0.005 0.01 0.1"
extension_Mymed=MyMedLt


acc_prefix=naeval
thresholdRelevance=4
thresholdAntiRelevance=2
cutoffs="5,10,20"
cutoffsWrite="5-10-20"



lambdas="0.1 0.2 0.5 0.7 1"

for dataset in $datasets
do
    for split in $splits
    do
        recommendation_folder="RecommendationFile"$split
        mkdir -p $recommendation_folder

        results_Folder="ResultFolder"$split
        mkdir -p $results_Folder


        train_file=TrainTest"$split"/"$dataset""$split"Train.txt
        test_file=TrainTest"$split"/"$dataset""$split"Test.txt

        output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_"SkylineTestOrder"_"$rec_strat".txt
        $java_command $jvm_memory -jar $JAR -o skylineRecommenders -trf $train_file -tsf $test_file -cIndex true -rr "SkylineTestOrder" -rs "notUsed" -n 20 -nI $items_recommended -orf $output_rec_file -recStrat $rec_strat



        for ranksys_recommender in PopularityRecommender RandomRecommender
        do

          output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_"$ranksys_recommender"_"$rec_strat".txt
          $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex true -rr "$ranksys_recommender" -rs "notUsed" -nI $items_recommended -n 20 -orf $output_rec_file -recStrat $rec_strat

          output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_"$ranksys_recommender"CF_"$rec_strat".txt
          $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr "$ranksys_recommender" -rs "notUsed" -nI $items_recommended -n 20 -orf $output_rec_file -recStrat $rec_strat

        done # End pop and Rnd
        wait




        for neighbours in $all_neighbours
        do


          #Ranksys with similarities UserBased
          for UB_sim in SetJaccardUserSimilarity VectorCosineUserSimilarity
          do
            output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_UB_"$UB_sim"_k"$neighbours"_"$rec_strat".txt
            $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr UserNeighborhoodRecommender -rs $UB_sim -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat


            #ClassicUBWithAntiSim - no lambdas - It is equivalent to nndiv
            for inv in true false
            do
                # sim - rdsupp
                output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_ClassicUBWithAntiSim_"inv""$inv"_"$UB_sim"_"AntiDifferenceRatings"_normTrue_k"$neighbours"_"$rec_strat"OPT.txt
                $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr ClassicUBWithAntiSim -rs $UB_sim -rs2 "AntiDifferenceRatingsPerUserSimilarity" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -applyJaccard true -inverse $inv

                 # sim - rat-diff
                 output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_ClassicUBWithAntiSim_"inv""$inv"_"$UB_sim"_"AntiDifferenceRatings"_normFalse_k"$neighbours"_"$rec_strat"OPT.txt
                 $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr ClassicUBWithAntiSim -rs $UB_sim -rs2 "AntiDifferenceRatingsPerUserSimilarity" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -applyJaccard false -inverse $inv

                 # sim - bin-sets
                 output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_ClassicUBWithAntiSim_"inv""$inv"_"$UB_sim"_"AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard"_k"$neighbours"_"$rec_strat"OPT.txt
                 $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr ClassicUBWithAntiSim -rs $UB_sim -rs2 "AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -thr 3 -inverse $inv
            done
            wait


            #UserAndAntiNeighborhoodRankingCombination - no lambdas - indr
            combiner=defaultcomb
            normalizer=stdnorm

            # sim - rdsupp
            output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_RankCombUBAndAntiUB_"$UB_sim"_"AntiDifferenceRatings"_normTrue_k"$neighbours"_com"$combiner"_n"$normalizer""$rec_strat"OPT.txt
            $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr RankCombUBAndAntiUB -rs $UB_sim -rs2 "AntiDifferenceRatingsPerUserSimilarity" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -applyJaccard true -normAgLib $normalizer -combAgLib $combiner

            # sim - rat-diff
            output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_RankCombUBAndAntiUB_"$UB_sim"_"AntiDifferenceRatings"_normFalse_k"$neighbours"_com"$combiner"_n"$normalizer""$rec_strat"OPT.txt
            $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr RankCombUBAndAntiUB -rs $UB_sim -rs2 "AntiDifferenceRatingsPerUserSimilarity" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -applyJaccard false -normAgLib $normalizer -combAgLib $combiner

            # sim - bin-sets
            output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_RankCombUBAndAntiUB_"$UB_sim"_"AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard"_k"$neighbours"_com"$combiner"_n"$normalizer""$rec_strat"OPT.txt
            $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr RankCombUBAndAntiUB -rs $UB_sim -rs2 "AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -thr 3 -normAgLib $normalizer -combAgLib $combiner




            #Our antiNeigh with the lambdas - inds

             # lambda is equivalent to theta in the formulation. (In this case, inv is equivalent TO TRUE)
             for l in $lambdas
                do

                    # sim - bin-sets
                    output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_UBAndAntiKNNRecommender_"$UB_sim"_"AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard"_lambda"$l"k"$neighbours"_"$rec_strat"OPT.txt
                    $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr UBAndAntiKNNRecommender -rs $UB_sim -rs2 "AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -thr 3 -lFactorizer $l

                    # sim - rdsupp
                    output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_UBAndAntiKNNRecommender_"$UB_sim"_"AntiDifferenceRatings"_normTrue_lambda"$l"k"$neighbours"_"$rec_strat"OPT.txt
                    $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr UBAndAntiKNNRecommender -rs $UB_sim -rs2 "AntiDifferenceRatingsPerUserSimilarity" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -applyJaccard true -lFactorizer $l

                    # sim - rat-diff
                    output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_UBAndAntiKNNRecommender_"$UB_sim"_"AntiDifferenceRatings"_normFalse_lambda"$l"k"$neighbours"_"$rec_strat"OPT.txt
                    $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr UBAndAntiKNNRecommender -rs $UB_sim -rs2 "AntiDifferenceRatingsPerUserSimilarity" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -applyJaccard false -lFactorizer $l

                    ##################



                done
                wait

                #Considering the weigh of the anti-neighs as positive (In this case, inv is equivalent TO FALSE)
                # lambda is equivalent to theta in the formulation
                for l in $lambdas
                do
                    # sim - bin-sets
                    output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_UBAndAntiKNNRecommender_"$UB_sim"_"AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard"_lambda"$l"k"$neighbours"_PosAnti"$rec_strat"OPT.txt
                    $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr UBAndAntiKNNRecommender -rs $UB_sim -rs2 "AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -thr 3 -lFactorizer $l -negAntiN false

                    # sim - rdsupp
                    output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_UBAndAntiKNNRecommender_"$UB_sim"_"AntiDifferenceRatings"_normTrue_lambda"$l"k"$neighbours"_PosAnti"$rec_strat"OPT.txt
                    $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr UBAndAntiKNNRecommender -rs $UB_sim -rs2 "AntiDifferenceRatingsPerUserSimilarity" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -applyJaccard true -lFactorizer $l -negAntiN false

                    # sim - rat-diff
                    output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_UBAndAntiKNNRecommender_"$UB_sim"_"AntiDifferenceRatings"_normFalse_lambda"$l"k"$neighbours"_PosAnti"$rec_strat"OPT.txt
                    $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr UBAndAntiKNNRecommender -rs $UB_sim -rs2 "AntiDifferenceRatingsPerUserSimilarity" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -applyJaccard false -lFactorizer $l -negAntiN false

                done # End UB sim
                wait


          done # End UB sim
          wait


          for IB_sim in SetJaccardItemSimilarity VectorCosineItemSimilarity
          do
            output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_IB_"$IB_sim"_k"$neighbours"_"$rec_strat".txt
            $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr ItemNeighborhoodRecommender -rs $IB_sim -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat


            for inv in true false
            do
                 # sim - rdsupp
                 output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_ClassicIBWithAntiSim_"inv""$inv"_"$IB_sim"_"AntiDifferenceRatings"_normTrue_k"$neighbours"_"$rec_strat"OPT.txt
                 $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr ClassicIBWithAntiSim -rs $IB_sim -rs2 "AntiDifferenceRatingsPerUserSimilarity" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -applyJaccard true -inverse $inv
                 sleep 1
                 # sim - rat-diff
                 output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_ClassicIBWithAntiSim_"inv""$inv"_"$IB_sim"_"AntiDifferenceRatings"_normFalse_k"$neighbours"_"$rec_strat"OPT.txt
                 $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr ClassicIBWithAntiSim -rs $IB_sim -rs2 "AntiDifferenceRatingsPerUserSimilarity" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -applyJaccard false -inverse $inv
                 sleep 1
                 # sim - bin-sets
                 output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_ClassicIBWithAntiSim_"inv""$inv"_"$IB_sim"_"AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard"_k"$neighbours"_"$rec_strat"OPT.txt
                 $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr ClassicIBWithAntiSim -rs $IB_sim -rs2 "AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -thr 3 -inverse $inv
            done
            wait


            #UserAndAntiNeighborhoodRankingCombination - no lambdas - indr

            combiner=defaultcomb
            normalizer=stdnorm

            # sim - rdsupp
            output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_RankCombIBAndAntiIB_"$IB_sim"_"AntiDifferenceRatings"_normTrue_k"$neighbours"_com"$combiner"_n"$normalizer""$rec_strat"OPT.txt
            $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr RankCombIBAndAntiIB -rs $IB_sim -rs2 "AntiDifferenceRatingsPerUserSimilarity" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -applyJaccard true -normAgLib $normalizer -combAgLib $combiner

            # sim - rat-diff
            output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_RankCombIBAndAntiIB_"$IB_sim"_"AntiDifferenceRatings"_normFalse_k"$neighbours"_com"$combiner"_n"$normalizer""$rec_strat"OPT.txt
            $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr RankCombIBAndAntiIB -rs $IB_sim -rs2 "AntiDifferenceRatingsPerUserSimilarity" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -applyJaccard false -normAgLib $normalizer -combAgLib $combiner

            # sim - bin-sets
            output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_RankCombIBAndAntiIB_"$IB_sim"_"AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard"_k"$neighbours"_com"$combiner"_n"$normalizer""$rec_strat"OPT.txt
            $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr RankCombIBAndAntiIB -rs $IB_sim -rs2 "AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -thr 3 -normAgLib $normalizer -combAgLib $combiner



            for l in $lambdas
                do

                    # sim - bin-sets
                    output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_IBAndAntiKNNRecommender_"$IB_sim"_"AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard"_lambda"$l"k"$neighbours"_"$rec_strat"OPT.txt
                    $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr IBAndAntiKNNRecommender -rs $IB_sim -rs2 "AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -thr 3 -lFactorizer $l

                    # sim - rdsupp
                    output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_IBAndAntiKNNRecommender_"$IB_sim"_"AntiDifferenceRatings"_normTrue_lambda"$l"k"$neighbours"_"$rec_strat"OPT.txt
                    $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr IBAndAntiKNNRecommender -rs $IB_sim -rs2 "AntiDifferenceRatingsPerUserSimilarity" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -applyJaccard true -lFactorizer $l

                    # sim - rat-diff
                    output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_IBAndAntiKNNRecommender_"$IB_sim"_"AntiDifferenceRatings"_normFalse_lambda"$l"k"$neighbours"_"$rec_strat"OPT.txt
                    $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr IBAndAntiKNNRecommender -rs $IB_sim -rs2 "AntiDifferenceRatingsPerUserSimilarity" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -applyJaccard false -lFactorizer $l



                done
                wait

                #pos anti
                for l in $lambdas
                do

                    # sim - bin-sets
                    output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_IBAndAntiKNNRecommender_"$IB_sim"_"AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard"_lambda"$l"k"$neighbours"_PosAntiFix"$rec_strat"OPT.txt
                    $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr IBAndAntiKNNRecommender -rs $IB_sim -rs2 "AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -thr 3 -lFactorizer $l -negAntiN false

                    # sim - rdsupp
                    output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_IBAndAntiKNNRecommender_"$IB_sim"_"AntiDifferenceRatings"_normTrue_lambda"$l"k"$neighbours"_PosAntiFix"$rec_strat"OPT.txt
                    $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr IBAndAntiKNNRecommender -rs $IB_sim -rs2 "AntiDifferenceRatingsPerUserSimilarity" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -applyJaccard true -lFactorizer $l -negAntiN false

                    # sim - rat-diff
                    output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_ranksys_IBAndAntiKNNRecommender_"$IB_sim"_"AntiDifferenceRatings"_normFalse_lambda"$l"k"$neighbours"_PosAntiFix"$rec_strat"OPT.txt
                    $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr IBAndAntiKNNRecommender -rs $IB_sim -rs2 "AntiDifferenceRatingsPerUserSimilarity" -nI $items_recommended -n $neighbours -orf $output_rec_file -recStrat $rec_strat -applyJaccard false -lFactorizer $l -negAntiN false



                done
                wait

          done
          wait #IB Sim


        done # End neighs
        wait


        for recommender in MFRecommenderHKV
        do
          for k_factor in $k_factors
          do
            for lambda_val in $lambda_factorizers
            do
              for alpha_val in $alpha_factorizers
              do
                #Neighbours is put to 20 because this recommender does not use it
                output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_"$recommender"_kFactor"$k_factor"_a"$alpha_val"_l"$lambda_val"_"$rec_strat".txt
                $java_command $jvm_memory -jar $JAR -o ranksysOnlyComplete -trf $train_file -tsf $test_file -cIndex false -rr $recommender -rs "notUsed" -nI $items_recommended -n 20 -orf $output_rec_file -kFactorizer $k_factor -aFactorizer $alpha_val -lFactorizer $lambda_val -recStrat $rec_strat
              done
              wait #End alpha values
            done
            wait #End lambda
          done
          wait #End KFactor
        done
        wait #End RankRecommender


        #BPR
        for repetition in 1 #2 3 4
        do
              for bpr_factor in $bpr_factors
              do
                for bpr_bias_reg in $bpr_bias_regs
                do
                  for bpr_reg_U in $bpr_regs_u #Regularization for items and users is the same
                  do
                    bpr_reg_J=$(echo "$bpr_reg_U/10" | bc -l)

                    output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_"$extension_Mymed"_BPRMF_nFact"$bpr_factor"_nIter"$bpr_iter"_LearnR"$bpr_learn_rate"_BiasR"$bpr_bias_reg"_RegU"$bpr_reg_U"_RegI"$bpr_reg_U"_RegJ"$bpr_reg_J""Rep$repetition"_"$rec_strat".txt
                    echo $output_rec_file
                    output_rec_file2=$output_rec_file"Aux".txt
                    if [ ! -f *"$output_rec_file"* ]; then
                        echo "./$mymedialite_path/item_recommendation --training-file=$train_file --recommender=BPRMF --prediction-file=$output_rec_file2 --predict-items-number=$items_recommended --recommender-options="num_factors=$bpr_factor bias_reg=$bpr_bias_reg reg_u=$bpr_reg_U reg_i=$bpr_reg_U reg_j=$bpr_reg_J learn_rate=$bpr_learn_rate UniformUserSampling=false WithReplacement=false num_iter=$bpr_iter""
                        ./$mymedialite_path/item_recommendation --training-file=$train_file --recommender=BPRMF --prediction-file=$output_rec_file2 --predict-items-number=$items_recommended --recommender-options="num_factors=$bpr_factor bias_reg=$bpr_bias_reg reg_u=$bpr_reg_U reg_i=$bpr_reg_U reg_j=$bpr_reg_J learn_rate=$bpr_learn_rate UniformUserSampling=false WithReplacement=false num_iter=$bpr_iter"
                        $java_command $jvm_memory -jar $JAR -o ParseMyMediaLite -trf $output_rec_file2 $test_file $output_rec_file
                        rm $output_rec_file2

                    fi
                  done # Reg U
                  wait
                done # Bias reg
                wait
              done # Factors
              wait

            done # Repetition
            wait

        lambda_easer=0.5
        output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_"easer"_"lambda"$lambda_easer"_ImplicitTrue"_"$rec_strat".txt
        python ease_rec/main.py --training $train_file --test $test_file --implicit True --lamb $lambda_easer --nI $items_recommended --result $output_rec_file

        output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_"easer"_"lambda"$lambda_easer"_ImplicitFalse"_"$rec_strat".txt
        python ease_rec/main.py --training $train_file --test $test_file --lamb $lambda_easer --nI $items_recommended --result $output_rec_file



        for beta in "0.6" "0.7"
        do
          for alpha in "1" "2"
          do

              output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_"RP3beta"_"beta"$beta"_"alpha""$alpha"_ImplicitTrue"_"$rec_strat".txt
              python recommender-systems/run2.py --training $train_file --test $test_file --nI $items_recommended --result $output_rec_file --implicit True --alpha $alpha --beta $beta

              output_rec_file=$recommendation_folder/"$rec_prefix"_"$dataset""$split"_"RP3beta"_"beta"$beta"_"alpha""$alpha"_ImplicitFalse"_"$rec_strat".txt
              python recommender-systems/run2.py --training $train_file --test $test_file --nI $items_recommended --result $output_rec_file --alpha $alpha --beta $beta

          done
          wait
        done # Repetition
        wait



        find $recommendation_folder/ -name "$rec_prefix"*"$dataset""$split"* | while read recFile; do
            rec_FileName=$(basename "$recFile" .txt) #extension removed

            if [[ $rec_FileName == *_* ]]; then
                # Basic Evaluation. Only Acc Metrics and NonAcc most basic


                output_result_file=$results_Folder/"$acc_prefix"_RelTh"$thresholdRelevance"AntiRelTh"$thresholdAntiRelevance"_"$rec_FileName"_C$cutoffsWrite".txt"
                $java_command $jvm_memory -jar $JAR -o ranksysNonAccuracyMetricsEvaluation -trf $train_file -tsf $test_file -rf $recFile -thr $thresholdRelevance -rc $cutoffs -orf $output_result_file -onlyAcc false

            fi
        done #End find
        wait





    done # end splits
    wait



done # End dataset
wait
