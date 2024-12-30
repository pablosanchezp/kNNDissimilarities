package es.uam.eps.knndissimilarities.sims;

import java.util.Set;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;

/***
 * THis is basically the AntiPositiveNegativeBinarySetItemsPerUserSimilarity but changing the denominator computation of the similarity. In this case:
 * 
 * @author Pablo Sanchez (pablo.sanchezp@uam.es)
 *
 */
public class AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard extends AntiPositiveNegativeBinarySetItemsPerUserSimilarity {

	public AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard(FastPreferenceData<?, ?> data, double threshold) {
		super(data, threshold);
	}
	
	@Override
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
		
		double denominator = unionPositiveNegative + unionNegativePositive;
		
		if (denominator == 0) {
			return 0;
		}
		
		return (intersecPositiveNegative + intersecNegativePositive) / denominator;
	}
	
	@Override
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
        	double denominator = (uidxPos + this.prefsNegativeIdxs.get(vidx).size() - (double) inserSecPositiveNegative.get(vidx.intValue())) + (uidxNeg + this.prefsPositiveIdxs.get(vidx).size() - (double) inserSecNegativePositive.get(vidx.intValue()));
        	productMap.addTo(vidx, (double) inserSecPositiveNegative.get(vidx.intValue()) / denominator);
        });
        
        inserSecNegativePositive.keySet().forEach(vidx -> {
        	double denominator = (uidxPos + this.prefsNegativeIdxs.get(vidx).size() - (double) inserSecPositiveNegative.get(vidx.intValue())) + (uidxNeg + this.prefsPositiveIdxs.get(vidx).size() - (double) inserSecNegativePositive.get(vidx.intValue()));

        	productMap.addTo(vidx, (double) inserSecNegativePositive.get(vidx.intValue()) / denominator );
        });

        return productMap;
    }
}
