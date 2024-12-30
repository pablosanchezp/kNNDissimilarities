package es.uam.eps.knndissimilarities.utils;





import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.CachedItemNeighborhood;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.ItemNeighborhood;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.TopKItemNeighborhood;
import es.uam.eps.ir.ranksys.nn.item.sim.SetCosineItemSimilarity;
import es.uam.eps.ir.ranksys.nn.item.sim.SetJaccardItemSimilarity;
import es.uam.eps.ir.ranksys.nn.item.sim.VectorCosineItemSimilarity;
import es.uam.eps.ir.ranksys.nn.item.sim.VectorJaccardItemSimilarity;
import es.uam.eps.ir.ranksys.nn.user.neighborhood.TopKUserNeighborhood;
import es.uam.eps.ir.ranksys.nn.user.neighborhood.UserNeighborhood;
import es.uam.eps.ir.ranksys.nn.user.sim.SetCosineUserSimilarity;
import es.uam.eps.ir.ranksys.nn.user.sim.SetJaccardUserSimilarity;
import es.uam.eps.ir.ranksys.nn.user.sim.VectorCosineUserSimilarity;
import es.uam.eps.ir.ranksys.nn.user.sim.VectorJaccardUserSimilarity;
import es.uam.eps.ir.ranksys.rec.Recommender;
import es.uam.eps.ir.ranksys.rec.fast.basic.PopularityRecommender;
import es.uam.eps.ir.ranksys.rec.fast.basic.RandomRecommender;
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

	                }
	                UserNeighborhood<Long> positiveUrneighborhood = new TopKUserNeighborhood<>(simUNRPositive, neighbours);
	                return new UserClassicNeighborhoodCombinedWithAntiSim<>(trainPrefData, positiveUrneighborhood, simUNRNegative, 1, inverse);

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

	                }
	                UserNeighborhood<Long> positiveUrneighborhood = new TopKUserNeighborhood<>(simUNRPositive, neighbours);
	                UserNeighborhood<Long> negativeUrneighborhood = new TopKUserNeighborhood<>(simUNRNegative, neighbours);

	                return new UserAndAntiNeighborhoodRankingCombination<>(trainPrefData, positiveUrneighborhood, negativeUrneighborhood, 1, normalizer, combiner);
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

}
