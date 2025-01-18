package es.uam.eps.knndissimilarities.utils;






import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Stream;

import org.jooq.lambda.tuple.Tuple3;

import es.uam.eps.ir.attrrec.datamodel.feature.UserFeatureData;
import es.uam.eps.ir.attrrec.metrics.recommendation.averages.WeightedModelUser.UserMetricWeight;

import es.uam.eps.ir.ranksys.core.preference.PreferenceData;
import es.uam.eps.ir.ranksys.core.preference.SimplePreferenceData;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.metrics.RecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.basic.AveragePrecision;
import es.uam.eps.ir.ranksys.metrics.basic.NDCG;
import es.uam.eps.ir.ranksys.metrics.basic.ReciprocalRank;
import es.uam.eps.ir.ranksys.metrics.rank.RankingDiscountModel;
import es.uam.eps.ir.ranksys.metrics.rel.BinaryRelevanceModel;
import es.uam.eps.ir.ranksys.metrics.rel.RelevanceModel;
import es.uam.eps.ir.ranksys.mf.Factorization;
import es.uam.eps.ir.ranksys.mf.als.HKVFactorizer;
import es.uam.eps.ir.ranksys.mf.rec.MFRecommender;
import es.uam.eps.ir.ranksys.nn.item.ItemNeighborhoodRecommender;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.CachedItemNeighborhood;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.ItemNeighborhood;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.TopKItemNeighborhood;
import es.uam.eps.ir.ranksys.nn.item.sim.SetCosineItemSimilarity;
import es.uam.eps.ir.ranksys.nn.item.sim.SetJaccardItemSimilarity;
import es.uam.eps.ir.ranksys.nn.item.sim.VectorCosineItemSimilarity;
import es.uam.eps.ir.ranksys.nn.item.sim.VectorJaccardItemSimilarity;
import es.uam.eps.ir.ranksys.nn.user.UserNeighborhoodRecommender;
import es.uam.eps.ir.ranksys.nn.user.neighborhood.TopKUserNeighborhood;
import es.uam.eps.ir.ranksys.nn.user.neighborhood.UserNeighborhood;
import es.uam.eps.ir.ranksys.nn.user.sim.SetCosineUserSimilarity;
import es.uam.eps.ir.ranksys.nn.user.sim.SetJaccardUserSimilarity;
import es.uam.eps.ir.ranksys.nn.user.sim.VectorCosineUserSimilarity;
import es.uam.eps.ir.ranksys.nn.user.sim.VectorJaccardUserSimilarity;
import es.uam.eps.ir.ranksys.novelty.longtail.FDItemNovelty;
import es.uam.eps.ir.ranksys.novelty.longtail.PCItemNovelty;
import es.uam.eps.ir.ranksys.novelty.longtail.metrics.EFD;
import es.uam.eps.ir.ranksys.novelty.longtail.metrics.EPC;

import es.uam.eps.ir.ranksys.rec.Recommender;
import es.uam.eps.ir.ranksys.rec.fast.basic.PopularityRecommender;
import es.uam.eps.ir.ranksys.rec.fast.basic.RandomRecommender;
import es.uam.eps.knndissimilarities.rec.ItemAndAntiNeighborhoodRankingCombination;
import es.uam.eps.knndissimilarities.rec.ItemNeighborhoodAndAntiNeighborhoodRecommender;
import es.uam.eps.knndissimilarities.rec.UserAndAntiNeighborhoodRankingCombination;
import es.uam.eps.knndissimilarities.rec.UserClassicNeighborhoodCombinedWithAntiSim;
import es.uam.eps.knndissimilarities.rec.UserNeighborhoodAndAntiNeighborhoodRecommender;
import es.uam.eps.knndissimilarities.sims.AntiDifferenceRatingsPerUserSimilarityItem;
import es.uam.eps.knndissimilarities.sims.AntiDifferenceRatingsPerUserSimilarityUser;
import es.uam.eps.knndissimilarities.sims.AntiPositiveNegativeBinarySetItemsPerUserSimilarityItem;
import es.uam.eps.knndissimilarities.sims.AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccardItem;
import es.uam.eps.knndissimilarities.sims.AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccardUser;



public class AntiNeighborsUtils {

	public static Recommender<Long, Long> obtRankSysRecommeder(String ranksysRecommender, String ranksysSimilarity, String ranksysSimilarity2,
			FastPreferenceData<Long, Long> trainPrefData, int neighbours, Integer numFactors, Double alphaFactorizer,
			Double lambdaFactorizer, Integer numIterations, boolean inverse, boolean applyJaccard, double threshold, String combiner, String normalizer, double lambda, boolean negAnti) {
		
		switch (ranksysRecommender) {
	     	case "RndRec":
	         case "RandomRecommender": {
	         	System.out.println("RandomRecommender");
	             return new RandomRecommender<>(trainPrefData, trainPrefData);
	         }
	         case "PopRec":
	         case "PopularityRecommender":{
	         	System.out.println("PopularityRecommender");
	             return new PopularityRecommender<>(trainPrefData);
	         }
	         case "MFRecHKV":
	            case "MFRecommenderHKV": { // Matrix factorization
	                int k = numFactors;
	                double alpha = alphaFactorizer;
	                int numIter = numIterations;
	                System.out.println("MFRecommenderHKV");
	                System.out.println("kFactors: " + k);
	                System.out.println("lambda: " + lambda);
	                System.out.println("alpha: " + alpha);
	                System.out.println("numIter: " + numIter);


	                DoubleUnaryOperator confidence = x -> 1 + alpha * x;
	                Factorization<Long, Long> factorization = new HKVFactorizer<Long, Long>(lambda, confidence, numIter)
	                        .factorize(k, trainPrefData);
	                return new MFRecommender<>(trainPrefData, trainPrefData, factorization);
	            }
	         case "UserNeighborhoodRecommender": { // User based. Pure CF recommendation
	                es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity<Long> simUNR = obtRanksysUserSimilarity(trainPrefData, ranksysSimilarity);
	                if (simUNR == null) {
	                    return null;
	                } else {
	                	System.out.println("UserNeighborhoodRecommender");
	                	System.out.println("kNeighs: "+ neighbours);
	                	System.out.println("Sim: "+ ranksysSimilarity);
	                }
	                UserNeighborhood<Long> urneighborhood = new TopKUserNeighborhood<>(simUNR, neighbours);
	                return new UserNeighborhoodRecommender<>(trainPrefData, urneighborhood, 1);
	            }
	         case "ItemNeighborhoodRecommender": {// Item based. Pure CF recommendation
	                es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity<Long> simINR = obtRanksysItemSimilarity(trainPrefData, ranksysSimilarity);
	                if (simINR == null) {
	                	return null;
	                }else{
	                	System.out.println("ItemNeighborhoodRecommender");
	                	System.out.println("kNeighs: "+ neighbours);
	                	System.out.println("Sim: "+ ranksysSimilarity);             
	                }
	                ItemNeighborhood<Long> neighborhood = new TopKItemNeighborhood<>(simINR, neighbours);
	                neighborhood = new CachedItemNeighborhood<>(neighborhood);
	                return new ItemNeighborhoodRecommender<>(trainPrefData, neighborhood, 1);
	            }
	         case "UserClassicNeighborhoodCombinedWithAntiSim":
	            case "ClassicUBWithAntiSim": {
	            	es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity<Long> simUNRPositive = obtRanksysUserSimilarity(trainPrefData, ranksysSimilarity);
	                es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity<Long> simUNRNegative = obtRanksysUserAntiSimilarity(trainPrefData, ranksysSimilarity2, applyJaccard, threshold);
	                if (simUNRNegative != null && simUNRPositive != null) {
	                	System.out.println("ClassicUBWithAntiSim");
	                	System.out.println("kNeighs: " + neighbours);
	                	System.out.println("Sim positive: " + ranksysSimilarity);
	                	System.out.println("Sim negative: " + ranksysSimilarity2);
	                	System.out.println("Negated: " + inverse);

	                } else {
	                	return null;
	                }
	                UserNeighborhood<Long> positiveUrneighborhood = new TopKUserNeighborhood<>(simUNRPositive, neighbours);
	                return new UserClassicNeighborhoodCombinedWithAntiSim<>(trainPrefData, positiveUrneighborhood, simUNRNegative, 1, inverse);

	            }
	            
	          //Version of IB combined with anti-sim 
	            case "ItemClassicNeighborhoodCombinedWithAntiSim":
	            case "ClassicIBWithAntiSim": {
	            	es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity<Long> simINRPositive = obtRanksysItemSimilarity(trainPrefData, ranksysSimilarity);
	                es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity<Long> simINRNegative = obtRanksysItemAntiSimilarity(trainPrefData, ranksysSimilarity2, applyJaccard, threshold);
	                if (simINRPositive != null && simINRNegative != null) {
	                	System.out.println("ClassicIBWithAntiSim");
	                	System.out.println("kNeighs: " + neighbours);
	                	System.out.println("Sim positive: " + ranksysSimilarity);
	                	System.out.println("Sim negative: " + ranksysSimilarity2);
	                	System.out.println("Negated: " + inverse);
	                	System.out.println("OPTIMIZED");

	                } else {
	                	return null;
	                }
	                

	                ItemNeighborhood<Long> positiveIrneighborhood = new TopKItemNeighborhood<>(simINRPositive, neighbours);
	                positiveIrneighborhood = new CachedItemNeighborhood<>(positiveIrneighborhood);
	                
	                ItemNeighborhood<Long> combinedNeighborhood = new ItemRestrictedCombinedCachedNeighborhood<>(trainPrefData.numItems(), positiveIrneighborhood, simINRNegative, inverse);
	                
	                
	                return new ItemNeighborhoodRecommender<>(trainPrefData, combinedNeighborhood, 1);
	            }
	            
	            case "UserAndAntiNeighborhoodRankingCombination":
	            case "RankCombUBAndAntiUB": {
	            	es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity<Long> simUNRPositive = obtRanksysUserSimilarity(trainPrefData, ranksysSimilarity);
	                es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity<Long> simUNRNegative = obtRanksysUserAntiSimilarity(trainPrefData, ranksysSimilarity2, applyJaccard, threshold);
	                if (simUNRNegative != null && simUNRPositive != null) {
	                	System.out.println("UserAndAntiNeighborhoodRankingCombination");
	                	System.out.println("kNeighs: " + neighbours);
	                	System.out.println("Sim positive: " + ranksysSimilarity);
	                	System.out.println("Sim negative: " + ranksysSimilarity2);
	                	System.out.println("Normalizer: " + normalizer);
	                	System.out.println("Combiner: " + combiner);

	                } else {
	                	return null;
	                }
	                UserNeighborhood<Long> positiveUrneighborhood = new TopKUserNeighborhood<>(simUNRPositive, neighbours);
	                UserNeighborhood<Long> negativeUrneighborhood = new TopKUserNeighborhood<>(simUNRNegative, neighbours);

	                return new UserAndAntiNeighborhoodRankingCombination<>(trainPrefData, positiveUrneighborhood, negativeUrneighborhood, 1, normalizer, combiner);
	            }	
	            
	            case "ItemAndAntiNeighborhoodRankingCombination":
	            case "RankCombIBAndAntiIB": {
	            	es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity<Long> simINRPositive = obtRanksysItemSimilarity(trainPrefData, ranksysSimilarity);
	                es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity<Long> simINRNegative = obtRanksysItemAntiSimilarity(trainPrefData, ranksysSimilarity2, applyJaccard, threshold);
	                if (simINRNegative != null && simINRPositive != null) {
	                	System.out.println("ItemAndAntiNeighborhoodRankingCombination");
	                	System.out.println("kNeighs: " + neighbours);
	                	System.out.println("Sim positive: " + ranksysSimilarity);
	                	System.out.println("Sim negative: " + ranksysSimilarity2);
	                	System.out.println("Normalizer: " + normalizer);
	                	System.out.println("Combiner: " + combiner);

	                } else {
	                	return null;
	                }
	                ItemNeighborhood<Long> positiveUrneighborhood = new TopKItemNeighborhood<>(simINRPositive, neighbours);
	                positiveUrneighborhood = new CachedItemNeighborhood<>(positiveUrneighborhood);

	                ItemNeighborhood<Long> negativeUrneighborhood = new TopKItemNeighborhood<>(simINRNegative, neighbours);
	                negativeUrneighborhood = new CachedItemNeighborhood<>(negativeUrneighborhood);


	                return new ItemAndAntiNeighborhoodRankingCombination<>(trainPrefData, positiveUrneighborhood, negativeUrneighborhood, 1, normalizer, combiner);
	            }
	            
	            
	            //Version with lambdas ()
	            case "UBAndAntiKNN":
	            case "UBAndAntiKNNRecommender":
	            {
	                es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity<Long> simUNRPositive = obtRanksysUserSimilarity(trainPrefData, ranksysSimilarity);
	                es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity<Long> simUNRNegative = obtRanksysUserAntiSimilarity(trainPrefData, ranksysSimilarity2, applyJaccard, threshold);
	                if (simUNRNegative != null && simUNRPositive != null) {
	                	System.out.println("UBAndAntiKNNRecommender");
	                	System.out.println("normalize (apply jaccard)");
	                	System.out.println("lambda: " + lambda);
	                	System.out.println("kNeighs: " + neighbours);
	                	System.out.println("Sim positive: " + ranksysSimilarity);
	                	System.out.println("Sim negative: " + ranksysSimilarity2);
	                	System.out.println("Negative anti neighs: " + negAnti);
	                } else {
	                	return null;
	                }
	                
	                
	                UserNeighborhood<Long> positiveUrneighborhood = new TopKUserNeighborhood<>(simUNRPositive, neighbours);
	                UserNeighborhood<Long> negativeUrneighborhood = new TopKUserNeighborhood<>(simUNRNegative, neighbours);

	                
	                return new UserNeighborhoodAndAntiNeighborhoodRecommender<>(trainPrefData, positiveUrneighborhood, negativeUrneighborhood, 1, lambda, negAnti);
	            }
	            case "IBAndAntiKNN":
	            case "IBAndAntiKNNRecommender":
	            {
	                es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity<Long> simUNRPositive = obtRanksysItemSimilarity(trainPrefData, ranksysSimilarity, true);
	                es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity<Long> simUNRNegative = obtRanksysItemAntiSimilarity(trainPrefData, ranksysSimilarity2, applyJaccard, threshold);
	                if (simUNRNegative != null && simUNRPositive != null) {
	                	System.out.println("IBAndAntiKNNRecommender");
	                	System.out.println("normalize (apply jaccard)");
	                	System.out.println("lambda: " + lambda);
	                	System.out.println("kNeighs: " + neighbours);
	                	System.out.println("Sim positive: " + ranksysSimilarity);
	                	System.out.println("Sim negative: " + ranksysSimilarity2);
	                	System.out.println("Negative anti neighs: " + negAnti);


	                } else {
	                	return null;
	                }
	                
	                
	                ItemNeighborhood<Long> positiveUrneighborhood = new TopKItemNeighborhood<>(simUNRPositive, neighbours);
	                positiveUrneighborhood = new CachedItemNeighborhood<>(positiveUrneighborhood);

	                ItemNeighborhood<Long> negativeUrneighborhood = new TopKItemNeighborhood<>(simUNRNegative, neighbours);
	                //Comment this if fail
	                negativeUrneighborhood = new CachedItemNeighborhood<>(negativeUrneighborhood);

	                
	                return new ItemNeighborhoodAndAntiNeighborhoodRecommender<>(trainPrefData, positiveUrneighborhood, negativeUrneighborhood, 1, lambda, negAnti);
	            }
	            

	            
	         
	         
	         
		 }
		
		
		return null;
	}
	
	public static <U, I> es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity<I> obtRanksysItemAntiSimilarity(FastPreferenceData<U, I> data, String similarity, boolean applyJaccard, double threshold) {
        switch (similarity) {
        case "AntiDifferenceRatingsPerUserSimilarity":
        	return new AntiDifferenceRatingsPerUserSimilarityItem<>(data, applyJaccard);
        case "AntiPositiveNegativeBinarySetItemsPerUserSimilarity":
        	return new AntiPositiveNegativeBinarySetItemsPerUserSimilarityItem<>(data, threshold);
        case "AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard":
        	return new AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccardItem<>(data, threshold);
        	
        default:
        	System.out.println("RankSys Anti user similarity is null");
            return null;
        }
    }
	
	public static <U, I, F> es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity<I> obtRanksysItemSimilarity(
            FastPreferenceData<U, I> data, String similarity){
    	return obtRanksysItemSimilarity(data, similarity, true);
    }
    /**
     * Obtain RankSys Item Similarity
     *
     * @param data the preference data used to compute the similarities
     * @param similarity the string identifier of the similarity
     * @return a RankSys item similarity model
     */
    public static <U, I, F> es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity<I> obtRanksysItemSimilarity(
            FastPreferenceData<U, I> data, String similarity, boolean dense) {
        switch (similarity) {
            case "VectorCosineItemSimilarity":
                // 0.5 to make it symmetrical.
                return new VectorCosineItemSimilarity<>(data, 0.5, dense);
            case "VectorJaccardItemSimilarity":
                return new VectorJaccardItemSimilarity<>(data, dense);
            case "SetJaccardItemSimilarity":
                return new SetJaccardItemSimilarity<>(data, dense);
            case "SetCosineItemSimilarity":
                return new SetCosineItemSimilarity<>(data, 0.5, dense);	
            default:
            	System.out.println(similarity + " not recognized");
                return null;
        }
    }
	
	public static <U, I> es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity<U> obtRanksysUserAntiSimilarity(FastPreferenceData<U, I> data, String similarity, boolean applyJaccard, double threshold) {
		switch (similarity) {
        case "AntiDifferenceRatingsPerUserSimilarity":
        	return new AntiDifferenceRatingsPerUserSimilarityUser<>(data, applyJaccard);
        case "AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard":
        	return new AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccardUser<>(data, threshold);
        	
        default:
        	System.out.println("RankSys Anti user similarity is null");
        	System.out.println("Check if we return a normal sim");
        	return obtRanksysUserSimilarity(data, similarity);        
        }
    }
	
	 public static <U, I> es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity<U> obtRanksysUserSimilarity(FastPreferenceData<U, I> data, String similarity) {
	        switch (similarity) {
	            case "VectorCosineUserSimilarity":
	                // 0.5 to make it symmetrical.
	                return new VectorCosineUserSimilarity<>(data, 0.5, true);
	            case "VectorJaccardUserSimilarity":
	                return new VectorJaccardUserSimilarity<>(data, true);
	            case "SetJaccardUserSimilarity":
	                return new SetJaccardUserSimilarity<>(data, true);
	            case "SetCosineUserSimilarity":
	                return new SetCosineUserSimilarity<>(data, 0.5, true);
	            default:
	            	System.out.println("RankSys user similarity is null");
	                return null;
	        }
	    }
	 
	 /***
	     * Method to filter the preference data by providing the set of valid users and valid items 
	     * @param original the original preferences
	     * @param validUsers the set of valid users
	     * @param validItems the set of valid items
	     * @return the preference data filtered
	     */
	    public static <U, I> PreferenceData<U, I> filterPreferenceData(PreferenceData<U, I> original, Set<U> validUsers, Set<I> validItems) {
	        final List<Tuple3<U, I, Double>> tuples = new ArrayList<>();
	        original.getUsersWithPreferences().filter(u -> validUsers.contains(u)).forEach(u -> {
	            if (validItems != null) {
	                original.getUserPreferences(u).filter(t -> (validItems.contains(t.v1))).forEach(idPref -> {
	                    tuples.add(new Tuple3<>(u, idPref.v1, idPref.v2));
	                });
	            } else {
	                original.getUserPreferences(u).forEach(idPref -> {
	                    tuples.add(new Tuple3<>(u, idPref.v1, idPref.v2));
	                });
	            }
	        });
	        System.out.println("Tuples original: " + original.numPreferences());
	        System.out.println("Tuples original filtered: " + tuples.size());
	        Stream<Tuple3<U, I, Double>> prev = tuples.stream();
	        PreferenceData<U, I> result = SimplePreferenceData.load(prev);
	        return result;
	    }
	    
	    public static void addMetrics(Map<String, RecommendationMetric<Long, Long>> recMetricsAvgRelUsers,
				Map<String, RecommendationMetric<Long, Long>> recMetricsAllRecUsers, int threshold, int cutoff,
				PreferenceData<Long, Long> trainData,
				PreferenceData<Long, Long> testData, RelevanceModel<Long, Long> selectedRelevance,
				BinaryRelevanceModel<Long, Long> binRel, RankingDiscountModel discModel, boolean computeOnlyAcc) throws IOException {

			recMetricsAvgRelUsers.put("Precision@" + cutoff + "_" + threshold,
					new es.uam.eps.ir.ranksys.metrics.basic.Precision<>(cutoff, binRel));
			recMetricsAvgRelUsers.put("MAP@" + cutoff + "_" + threshold, new AveragePrecision<>(cutoff, binRel));
			recMetricsAvgRelUsers.put("Recall@" + cutoff + "_" + threshold,
					new es.uam.eps.ir.ranksys.metrics.basic.Recall<>(cutoff, binRel));
			recMetricsAvgRelUsers.put("MRR@" + cutoff + "_" + threshold, new ReciprocalRank<>(cutoff, binRel));
			recMetricsAvgRelUsers.put("NDCG@" + cutoff + "_" + threshold,
					new NDCG<>(cutoff, new NDCG.NDCGRelevanceModel<>(false, testData, threshold)));

			if (!computeOnlyAcc) {
				recMetricsAllRecUsers.put("epc@" + cutoff,
						new EPC<>(cutoff, new PCItemNovelty<>(trainData), selectedRelevance, discModel));
				recMetricsAllRecUsers.put("efd@" + cutoff,
						new EFD<>(cutoff, new FDItemNovelty<>(trainData), selectedRelevance, discModel));
			}

		}
	    
	    /***
		 * Method to get the user model weight
		 * @param userModelWeight
		 * @return the user model weight
		 */
		public static UserMetricWeight obtUserMetricWeight (String userModelWeight) {
			for (UserMetricWeight m : UserMetricWeight.values()) {
				if (m.toString().equals(userModelWeight) || m.getShortName().equals(userModelWeight) ) {
					return m;
				}
			}
			return null;
		}
		
		public static<U, F> boolean isValidUser(U user, Boolean computeFilter,  UserFeatureData<U, F, Double> userFeature, F feature) {
			if (userFeature == null || feature == null || !computeFilter) {
				return true;
			}
			
			return userFeature.getUserFeatures(user).map(tuples -> tuples.v1).anyMatch(f-> f.equals(feature));
		}

}
