package es.uam.eps.knndissimilarities.rec;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.ItemNeighborhood;
import es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity;
import es.uam.eps.ir.ranksys.nn.user.neighborhood.UserNeighborhood;
import es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity;
import es.uam.eps.ir.ranksys.rec.fast.FastRankingRecommender;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import static java.lang.Math.pow;

/***
 * Recommender that will combine the positive similarity of a pure user neighbor with an anti-similarity 
 * (adding both of them and dividing it by 2) in order to generate recommendations 
 * 
 * @author Pablo Sanchez (pablo.sanchezp@uam.es)
 *
 * @param <U>
 * @param <I>
 */
public class ItemClassicNeighborhoodCombinedWithAntiSim<U,I> extends FastRankingRecommender<U, I> {
	
	/**
     * Preference data.
     */
    protected final FastPreferenceData<U, I> data;

    /**
     * User (positive) neighborhood.
     */
    protected final ItemNeighborhood<I> positiveNeighborhood;
    
    protected final ItemSimilarity<I> itemAntiSim;

    protected final int q;
    
    protected final boolean inverse;

	public ItemClassicNeighborhoodCombinedWithAntiSim(FastPreferenceData<U, I> data, ItemNeighborhood<I> positiveNeighborhood, ItemSimilarity<I> antiSimilarity, int q, boolean inverse) {
		super(data, data);
		this.data = data;
		this.positiveNeighborhood = positiveNeighborhood;
		this.itemAntiSim = antiSimilarity;
		this.q = q;
		this.inverse = inverse;
		
	}



	@Override
	public Int2DoubleMap getScoresMap(int uidx) {
		Int2DoubleOpenHashMap scoresMap = new Int2DoubleOpenHashMap();
        scoresMap.defaultReturnValue(0.0);
        
        data.getUidxPreferences(uidx).forEach(jp -> positiveNeighborhood.getNeighbors(jp.v1)
                .forEach(is -> {
                    double w = pow(is.v2, q);
                    double antiw = this.itemAntiSim.similarity(jp.v1, is.v1);
                    if (this.inverse)
                    	antiw = - antiw;
                    
                    double finalw = (antiw + w);
                    
                    scoresMap.addTo(is.v1, finalw * jp.v2);
                }));

        return scoresMap;

	}
}
