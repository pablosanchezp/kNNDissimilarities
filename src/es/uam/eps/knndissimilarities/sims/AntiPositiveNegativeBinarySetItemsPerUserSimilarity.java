package es.uam.eps.knndissimilarities.sims;

import static org.ranksys.core.util.tuples.Tuples.tuple;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.function.IntToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.ranksys.core.util.tuples.Tuple2id;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.nn.sim.Similarity;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;


/***
 * Anti similarity defined with the following formula
 * 
 * Sim = ((I_u^+ intersec I_v^-) / (I_u^+ union I_v^-) + (I_u^- intersec I_v^+) / (I_u^- union I_v^+) / 2.0)
 * 
 * 
 * I_u^+ denotes the items rated by user u with a positive threshold
 * 
 * @author Pablo Sanchez
 *
 */
public class AntiPositiveNegativeBinarySetItemsPerUserSimilarity implements Similarity {
	protected final FastPreferenceData<?, ?> data;
	protected final double threshold;
	
	protected Map<Integer, Set<Integer>> prefsPositiveIdxs;
	protected Map<Integer, Set<Integer>> prefsNegativeIdxs;
	

	public AntiPositiveNegativeBinarySetItemsPerUserSimilarity(FastPreferenceData<?, ?> data, double threshold) {
		this.data = data;
		this.prefsPositiveIdxs = new HashMap<>();
		this.prefsNegativeIdxs = new HashMap<>();
		this.threshold = threshold;
		System.out.println("AntiPositiveNegativeBinarySetItemsPerUserSimilarity + OPT");
		System.out.println("Threshold: " + threshold);


	    this.data.getUidxWithPreferences().forEach(uidx -> {
	    	this.prefsPositiveIdxs.put(uidx, this.data.getUidxPreferences(uidx).filter(pref -> pref.v2 > this.threshold).mapToInt(t -> t.v1).boxed().collect(Collectors.toSet()));
	    	this.prefsNegativeIdxs.put(uidx, this.data.getUidxPreferences(uidx).filter(pref -> pref.v2 < this.threshold).mapToInt(t -> t.v1).boxed().collect(Collectors.toSet()));
	    });
	}


	
	protected double similarityBetween(Set<Integer> positivePrefs1, Set<Integer> negativePrefs1, Set<Integer> positivePrefs2, Set<Integer> negativePrefs2) {
		
		
		double unionPositiveNegative = positivePrefs1.size();
		double unionNegativePositive = negativePrefs1.size();
		
		double unionPositiveNegativeAux = negativePrefs2.size();
		double unionNegativePositiveAux  = positivePrefs2.size();
		
		
		
		double intersecPositiveNegative = 0;
		double intersecNegativePositive = 0;
		
		for (Integer iidx: positivePrefs1) {
			if (negativePrefs2.contains(iidx)) {
				intersecPositiveNegative++;
				unionPositiveNegativeAux--;
			}
		}
		
		for (Integer iidx: negativePrefs1) {
			if (positivePrefs2.contains(iidx)) {
				intersecNegativePositive++;
				unionNegativePositiveAux--;
			}
		}

		unionPositiveNegative += unionPositiveNegativeAux;
		unionNegativePositive += unionNegativePositiveAux;
		
		double pos = unionPositiveNegative == 0 ? 0 : intersecPositiveNegative/unionPositiveNegative;
		double neg = unionNegativePositive == 0 ? 0 : intersecNegativePositive/unionNegativePositive;
		
		return (pos + neg) / 2.0;
		
	}

	@Override
	public IntToDoubleFunction similarity(int idx) {
		return idx2 -> {
			return similarityBetween(this.prefsPositiveIdxs.get(idx), this.prefsNegativeIdxs.get(idx), this.prefsPositiveIdxs.get(idx2), this.prefsNegativeIdxs.get(idx2));
			};
	}

	protected Int2DoubleMap getProductMap(int uidx) {
        Int2DoubleOpenHashMap productMap = new Int2DoubleOpenHashMap();
        productMap.defaultReturnValue(0.0);
        
        
        Int2IntOpenHashMap inserSecPositiveNegative = new Int2IntOpenHashMap();
        inserSecPositiveNegative.defaultReturnValue(0);
        
        Int2IntOpenHashMap inserSecNegativePositive = new Int2IntOpenHashMap();
        inserSecNegativePositive.defaultReturnValue(0);
        

        this.prefsPositiveIdxs.get(uidx).forEach(i -> {
        	data.getIidxPreferences(i).filter(pref -> this.prefsNegativeIdxs.get(pref.v1).contains(i)).forEach(pref -> {
        		inserSecPositiveNegative.addTo(pref.v1, 1);
        	});
	    });
        this.prefsNegativeIdxs.get(uidx).forEach(i -> {
        	data.getIidxPreferences(i).filter(pref -> this.prefsPositiveIdxs.get(pref.v1).contains(i)).forEach(pref -> {
        		inserSecNegativePositive.addTo(pref.v1, 1);
        	});
	    });
        
        int uidxPos = this.prefsPositiveIdxs.get(uidx).size();
        int uidxNeg = this.prefsNegativeIdxs.get(uidx).size();

        inserSecPositiveNegative.keySet().forEach(vidx -> {
        	productMap.addTo(vidx, 0.5 * (double) inserSecPositiveNegative.get(vidx) / (uidxPos + this.prefsNegativeIdxs.get(vidx).size() - (double) inserSecPositiveNegative.get(vidx)) );
        });
        
        inserSecNegativePositive.keySet().forEach(vidx -> {
        	productMap.addTo(vidx, 0.5 * (double) inserSecNegativePositive.get(vidx) / (uidxNeg + this.prefsPositiveIdxs.get(vidx).size() - (double) inserSecNegativePositive.get(vidx)) );
        });

        return productMap;
    }


	@Override
	public Stream<Tuple2id> similarElems(int idx) {
		return getProductMap(idx).int2DoubleEntrySet().stream()
                .map(e -> {
                    int idx2 = e.getIntKey();
                    return tuple(idx2, e.getDoubleValue());
                });
	}

	
}
