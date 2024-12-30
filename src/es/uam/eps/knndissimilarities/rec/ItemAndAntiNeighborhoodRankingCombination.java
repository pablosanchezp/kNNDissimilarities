package es.uam.eps.knndissimilarities.rec;

import static java.lang.Math.pow;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.IntPredicate;
import java.util.stream.Collectors;

import org.ranksys.core.util.tuples.Tuple2id;

import es.uam.eps.ir.crossdomainPOI.utils.PreferenceComparators;
import es.uam.eps.ir.crossdomainPOI.utils.SequentialRecommendersUtils;
import es.uam.eps.ir.ranksys.fast.FastRecommendation;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.ItemNeighborhood;
import es.uam.eps.ir.ranksys.nn.user.neighborhood.UserNeighborhood;
import es.uam.eps.ir.ranksys.rec.fast.AbstractFastRecommender;
import es.uam.eps.nets.rankfusion.GenericRankAggregator;
import es.uam.eps.nets.rankfusion.GenericResource;
import es.uam.eps.nets.rankfusion.GenericSearchResults;
import es.uam.eps.nets.rankfusion.interfaces.IFCombiner;
import es.uam.eps.nets.rankfusion.interfaces.IFNormalizer;
import es.uam.eps.nets.rankfusion.interfaces.IFRankAggregator;
import es.uam.eps.nets.rankfusion.interfaces.IFResource;
import es.uam.eps.nets.rankfusion.interfaces.IFSearchResults;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;

/**
 * User neighborhood recommender that will combine two different rankings, the one with a negative neighborhood and the one with a positive one
 * @author Pablo Sanchez
 *
 * @param <U>
 * @param <I>
 */
public class ItemAndAntiNeighborhoodRankingCombination<U,I> extends AbstractFastRecommender<U, I>{
	
    protected final ItemNeighborhood<I> positiveNeighborhood;

    protected final ItemNeighborhood<I> negativeNeighborhood;
    
    protected final FastPreferenceData<U, I> data;

    protected final int q;
    
	private final IFRankAggregator rankAggregator;


	public ItemAndAntiNeighborhoodRankingCombination(FastPreferenceData<U, I> data, ItemNeighborhood<I> positiveNeighborhood, ItemNeighborhood<I> negativeNeighborhood, int q, String normalizer, String combiner) {
		super(data, data);
		System.out.println("ItemAndAntiNeighborhoodRankingCombination + Fix bug");

		this.positiveNeighborhood = positiveNeighborhood;
		this.negativeNeighborhood = negativeNeighborhood;
		this.data = data;
		this.q = q;
		
		IFNormalizer norm = SequentialRecommendersUtils.getNormalizer(normalizer, null, null);
		IFCombiner comb = SequentialRecommendersUtils.getCombiner(combiner);
		rankAggregator = new GenericRankAggregator(norm, comb);
	}

	@Override
	public FastRecommendation getRecommendation(int uidx, int maxLength, IntPredicate filter) {
		Int2DoubleMap mappingPositive = getScoresMapPositive(uidx, filter);
		Int2DoubleMap mappingNegative = getScoresMapNegative(uidx, filter);
		List<Tuple2id> lstPositives = mappingPositive.keySet().stream().map(iidx -> new Tuple2id(iidx, mappingPositive.get(iidx))).sorted(PreferenceComparators.recommendationComparatorTuple2id.reversed()).collect(Collectors.toList());
		List<Tuple2id> lstNegatives = mappingNegative.keySet().stream().map(iidx -> new Tuple2id(iidx, mappingNegative.get(iidx))).sorted(PreferenceComparators.recommendationComparatorTuple2id.reversed()).collect(Collectors.toList());

		
		List<IFSearchResults> neighboursList = new ArrayList<IFSearchResults>();
		Map<Long, IFResource> positiveNeighboursRanking = new HashMap<>();
		for (int i = 0; i < lstPositives.size(); i++) {
			String iidxS = Integer.toString(lstPositives.get(i).v1);
			
			positiveNeighboursRanking.put((long) iidxS.hashCode(), new GenericResource(iidxS, lstPositives.get(i).v2, i + 1));
		}
		
		double positiveNeighbourWeights = 1;
		IFSearchResults positiveNeighbourResults = new GenericSearchResults(positiveNeighboursRanking, positiveNeighbourWeights);
		positiveNeighbourResults.setNotRetrievedNormalizedValue(0.0);
		
		neighboursList.add(positiveNeighbourResults);

		
		Map<Long, IFResource> negativeNeighboursRanking = new HashMap<>();

		for (int i = 0; i < lstNegatives.size(); i++) {
			String iidxS = Integer.toString(lstNegatives.get(i).v1);
			
			negativeNeighboursRanking.put((long) iidxS.hashCode(), new GenericResource(iidxS, lstNegatives.get(i).v2, i + 1));
		}
		
		double negativeNeighbourWeights = 1;
		IFSearchResults negativeNeighbourResults = new GenericSearchResults(negativeNeighboursRanking, negativeNeighbourWeights);
		negativeNeighbourResults.setNotRetrievedNormalizedValue(0.0);
		
		neighboursList.add(negativeNeighbourResults);
		
		List<Tuple2id> items = new ArrayList<>();

		if (rankAggregator != null && neighboursList != null && neighboursList.size() != 0) {
			List<IFResource> aggResults = rankAggregator.aggregate(neighboursList)
					.getSortedResourceList(GenericSearchResults.I_RESOURCE_ORDER_COMBINED_VALUE);
			for (IFResource aggResult : aggResults) {
				// System.out.println(aggResult.getId() + " - " + aggResult.getCombinedValue());
				int iidx = Integer.parseInt(aggResult.getId());
				items.add(new Tuple2id(iidx, aggResult.getCombinedValue()));
			}
		}

		// It should be ordered
		Collections.sort(items, PreferenceComparators.recommendationComparatorTuple2id.reversed());
		// Avoid IndexOutOfBoundException
		items = items.subList(0, (maxLength > items.size()) ? items.size() : maxLength);
		return new FastRecommendation(uidx, items);
	}
	

	public Int2DoubleMap getScoresMapPositive(int uidx, IntPredicate filter) {
		Int2DoubleOpenHashMap scoresMap = new Int2DoubleOpenHashMap();
        scoresMap.defaultReturnValue(0.0);
        data.getUidxPreferences(uidx)
                .forEach(jp -> positiveNeighborhood.getNeighbors(jp.v1)
                        .forEach(is -> {
                        	if (filter.test(is.v1)) {
                        		double w = pow(is.v2, q);
                        		scoresMap.addTo(is.v1, w * jp.v2);
                        	}
                        }));

        return scoresMap;
	}
	
	public Int2DoubleMap getScoresMapNegative(int uidx, IntPredicate filter) {
		Int2DoubleOpenHashMap scoresMap = new Int2DoubleOpenHashMap();
        scoresMap.defaultReturnValue(0.0);
        data.getUidxPreferences(uidx)
                .forEach(jp -> negativeNeighborhood.getNeighbors(jp.v1)
                        .forEach(is -> {
                        	if (filter.test(is.v1)) {
                        		double w = pow(is.v2, q);
                        		scoresMap.addTo(is.v1, w * jp.v2);
                        	}
                        }));

        return scoresMap;
	}
}
