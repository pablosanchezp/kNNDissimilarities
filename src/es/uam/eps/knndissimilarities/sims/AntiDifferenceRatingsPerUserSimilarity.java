package es.uam.eps.knndissimilarities.sims;

import static org.ranksys.core.util.tuples.Tuples.tuple;

import java.util.HashMap;
import java.util.Map;
import java.util.function.IntToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.ranksys.core.util.tuples.Tuple2id;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import es.uam.eps.ir.ranksys.nn.sim.Similarity;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;

/***
 * 
 * The similarity is computed by the difference in the number of checkins that both users have rated
 * 
 * @author Pablo Sanchez (pablo.sanchezp@uam.es)
 *
 */
public class AntiDifferenceRatingsPerUserSimilarity implements Similarity {
	
	
	protected final FastPreferenceData<?, ?> data;
	protected final boolean applyJaccard;
	
	private Map<Integer, Map<Integer, Double>> prefsIdxs;
	private Int2IntOpenHashMap counters;
	
	public AntiDifferenceRatingsPerUserSimilarity(FastPreferenceData<?, ?> data, boolean applyJaccard) {
		this.data = data;
		this.applyJaccard = applyJaccard;
		this.prefsIdxs = new HashMap<>();
		System.out.println("AntiDifferenceRatingsPerUserSimilarity");
		System.out.println("Apply Jaccard: " + this.applyJaccard);
		System.out.println("OPT ");

		this.counters = new Int2IntOpenHashMap();
	    this.counters.defaultReturnValue(0);

	    this.data.getUidxWithPreferences().forEach(uidx -> counters.put(uidx, (int) this.data.getUidxPreferences(uidx).count()));
	    this.data.getUidxWithPreferences().forEach(uidx -> prefsIdxs.put(uidx, this.data.getUidxPreferences(uidx).collect(Collectors.toMap(IdxPref::v1, IdxPref::v2))));
	}


	
	private double similarityBetween(Map<Integer, Double> prefsIdx, Map<Integer, Double> prefsIdx2) {
		
		double accDiff = 0;
		double count = 0;
		
		int union = prefsIdx.size(); 
		int unionAux = prefsIdx2.size();
		
		for (Integer iidx: prefsIdx.keySet()) {
			double scoreIdx1 = prefsIdx.get(iidx);
			
			Double scoreIdx2 = prefsIdx2.get(iidx);
			if (scoreIdx2 != null) {
				
				//1 is to avoid numerical errors a value of prefsIdx.size is obtained if there is no difference in the ratings
				accDiff += 1.0/ (Math.abs(scoreIdx1 - scoreIdx2) + 1);
				count ++;
				unionAux--;
			}
		}
		
		//If there is nothing in common, we return 0
		if (count == 0)
			return 0;
		
		union +=unionAux;
		
		//We also multiply by the jaccard index because the higher the number of item in common rated differentely, the worst
		double finalMult = this.applyJaccard ? count/union : 1.0;
		
		return (1.0 - accDiff/count) * finalMult;
	}

	@Override
	public IntToDoubleFunction similarity(int idx) {
		
		return idx2 -> {
			return similarityBetween(getPrefs(idx), getPrefs(idx2));
			};
	}

	/*
	private Int2DoubleMap getProductMap(int idx1) {
        Int2DoubleOpenHashMap productMap = new Int2DoubleOpenHashMap();
        productMap.defaultReturnValue(0.0);
        

        ////
        Map<Integer, Double> prefsIdx = getPrefs(idx1);
        
        //If they have rated at least 1 common element we compute the similarities
        data.getUidxWithPreferences().filter(idx2 -> {
        	if (idx2 == idx1)
        		return false;
        	
        	Map<Integer, Double> prefsIdx2 = getPrefs(idx2);
        	for (Integer k: prefsIdx.keySet()) {
        		if (prefsIdx2.containsKey(k))
        			return true;
        	}
        	return false;
        }).forEach(uidx -> {
        	productMap.put(uidx, similarityBetween(prefsIdx, getPrefs(uidx)));
        });
        

        return productMap;
    }
	*/
	private Int2DoubleMap getProductMap(int uidx) {

	    Int2DoubleOpenHashMap productMap = new Int2DoubleOpenHashMap();
	    productMap.defaultReturnValue(0.0);
	    
	    Int2IntOpenHashMap counter = new Int2IntOpenHashMap();
	    counter.defaultReturnValue(0);


	    data.getUidxPreferences(uidx).forEach(i ->
	        data.getIidxPreferences(i.v1).forEach(v ->
	        {
	            if(uidx != v.v1)
	            {
	                productMap.addTo(v.v1, 1.0/ (Math.abs(i.v2 - v.v2) + 1));
	                counter.addTo(v.v1, 1);
	            };
	        }));

	    Int2DoubleMap definitive = new Int2DoubleOpenHashMap();
	    productMap.int2DoubleEntrySet().stream().forEach(entry ->
	    {
	        int vidx = entry.getIntKey();
	        double sim = entry.getDoubleValue();
	        double inter = counter.get(vidx);
	        double jaccard = applyJaccard ? (inter + 0.0) / (counters.get(uidx) + counters.get(vidx) - inter + 0.0) : 1.0;
	        definitive.put(vidx, (1.0 - sim/(inter + 0.0)) * jaccard);
	    });

	    return definitive;
	}
	

	@Override
	public Stream<Tuple2id> similarElems(int idx) {
		return getProductMap(idx).int2DoubleEntrySet().stream()
                .map(e -> {
                    int idx2 = e.getIntKey();
                    return tuple(idx2, e.getDoubleValue());
                });
	}

	private Map<Integer, Double> getPrefs(int idx){
		Map<Integer, Double> prefsIdx = null;
        if (prefsIdxs.containsKey(idx)) {
        	prefsIdx = this.prefsIdxs.get(idx);
        }
        return prefsIdx;
	}
}
