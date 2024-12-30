package es.uam.eps.knndissimilarities.rec;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
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
public class UserClassicNeighborhoodCombinedWithAntiSim<U,I> extends FastRankingRecommender<U, I> {
	
	/**
     * Preference data.
     */
    protected final FastPreferenceData<U, I> data;

    /**
     * User (positive) neighborhood.
     */
    protected final UserNeighborhood<U> positiveNeighborhood;
    
    protected final UserSimilarity<U> userAntiSim;

    protected final int q;
    
    protected final boolean inverse;

	public UserClassicNeighborhoodCombinedWithAntiSim(FastPreferenceData<U, I> data, UserNeighborhood<U> positiveNeighborhood, UserSimilarity<U> antiSimilarity, int q, boolean inverse) {
		super(data, data);
		this.data = data;
		this.positiveNeighborhood = positiveNeighborhood;
		this.userAntiSim = antiSimilarity;
		this.q = q;
		this.inverse = inverse;
		
	}



	@Override
	public Int2DoubleMap getScoresMap(int uidx) {
		Int2DoubleOpenHashMap scoresMap = new Int2DoubleOpenHashMap();
        scoresMap.defaultReturnValue(0.0);
        positiveNeighborhood.getNeighbors(uidx).forEach(vs -> {
            double w = pow(vs.v2, q);
            
            //higher values in the anti-sim means higher difference in the users, so i and 1 -
            double antiw = this.userAntiSim.similarity(uidx, vs.v1);
            if (this.inverse)
            	antiw = -antiw;
            	
            
            //the final weight would be the positive sim + the negated negative sim / 2
            double finalw = (antiw + w);
            
            data.getUidxPreferences(vs.v1).forEach(iv -> {
                double p = finalw * iv.v2;
                scoresMap.addTo(iv.v1, p);
            });
        });

        return scoresMap;
	}

}
