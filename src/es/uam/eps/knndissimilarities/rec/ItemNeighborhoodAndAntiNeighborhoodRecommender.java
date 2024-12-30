package es.uam.eps.knndissimilarities.rec;

import static java.lang.Math.pow;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.ItemNeighborhood;
import es.uam.eps.ir.ranksys.rec.fast.FastRankingRecommender;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;

public class ItemNeighborhoodAndAntiNeighborhoodRecommender<U,I> extends FastRankingRecommender<U, I> {

	/**
     * Preference data.
     */
    protected final FastPreferenceData<U, I> data;

    /**
     * User (positive) neighborhood.
     */ 
    protected final ItemNeighborhood<I> positiveNeighborhood;
    
    /**
     * User (negative) neighborhood.
     */
    protected final ItemNeighborhood<I> negativeNeighborhood;


    /**
     * Exponent of the similarity.
     */
    protected final int q;
    
    protected final double lambdaSim;

    protected final boolean negative;
	
	public ItemNeighborhoodAndAntiNeighborhoodRecommender(FastPreferenceData<U, I> data, ItemNeighborhood<I> positiveNeighborhood, ItemNeighborhood<I> negativeNeighborhood, int q, double lambdaSim, boolean negative) {
		super(data, data);
		this.data = data;
		this.q = q;
		this.positiveNeighborhood = positiveNeighborhood;
		this.negativeNeighborhood = negativeNeighborhood;
		this.lambdaSim = lambdaSim; //penalization for the anti-neighs
		this.negative = negative;
		System.out.println("Bug fixed");
	}

	@Override
	public Int2DoubleMap getScoresMap(int uidx) {
		Int2DoubleOpenHashMap scoresMap = new Int2DoubleOpenHashMap();
        scoresMap.defaultReturnValue(0.0);
        
        data.getUidxPreferences(uidx).forEach(jp -> {
            //positive neighs

        	positiveNeighborhood.getNeighbors(jp.v1).forEach(is -> {
                double w = pow(is.v2, q);
                scoresMap.addTo(is.v1, w * jp.v2);
            });
        	
            //negative neighs
        	negativeNeighborhood.getNeighbors(jp.v1).forEach(is -> {
                double w = pow(is.v2, q);
                if (this.negative) {
                	scoresMap.addTo(is.v1, -(w * jp.v2 * lambdaSim));
                } else {
                    scoresMap.addTo(is.v1, (w * jp.v2 * lambdaSim));
                }
            });

        });


        
        
        return scoresMap;
	}

}