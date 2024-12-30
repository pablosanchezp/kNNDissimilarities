package es.uam.eps.knndissimilarities.rec;

import static java.lang.Math.pow;

import java.util.List;
import java.util.stream.Collectors;

import org.ranksys.core.util.tuples.Tuple2id;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.nn.user.neighborhood.UserNeighborhood;
import es.uam.eps.ir.ranksys.rec.fast.FastRankingRecommender;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;


/***
 * Neighborhood recommender that combines 2 different neighborhoods, the positive and the negative one. The negative one will contribute negatively in the final score
 *
 * @author Pablo Sánchez <pablo.sanchezp@uam.es>
 *
 * @param <U>
 * @param <I>
 */
public class UserNeighborhoodAndAntiNeighborhoodRecommender<U,I> extends FastRankingRecommender<U, I> {

	/**
     * Preference data.
     */
    protected final FastPreferenceData<U, I> data;

    /**
     * User (positive) neighborhood.
     */
    protected final UserNeighborhood<U> positiveNeighborhood;
    
    /**
     * User (negative) neighborhood.
     */
    protected final UserNeighborhood<U> negativeNeighborhood;


    /**
     * Exponent of the similarity.
     */
    protected final int q;
    
    protected final double lambdaSim;
	
    protected final boolean negative;
    
	public UserNeighborhoodAndAntiNeighborhoodRecommender(FastPreferenceData<U, I> data, UserNeighborhood<U> positiveNeighborhood, UserNeighborhood<U> negativeNeighborhood, int q, double lambdaSim, boolean negative) {
		super(data, data);
		this.data = data;
		this.q = q;
		this.positiveNeighborhood = positiveNeighborhood;
		this.negativeNeighborhood = negativeNeighborhood;
		this.lambdaSim = lambdaSim; //penalization for the anti-neighs
		this.negative = negative;
		System.out.println("Negative: " + this.negative);
		System.out.println("Bug fixed");
	}

	@Override
	public Int2DoubleMap getScoresMap(int uidx) {
		Int2DoubleOpenHashMap scoresMap = new Int2DoubleOpenHashMap();
        scoresMap.defaultReturnValue(0.0);
        
        
        positiveNeighborhood.getNeighbors(uidx).forEach(vs -> {
            double w = pow(vs.v2, q);
            data.getUidxPreferences(vs.v1).forEach(iv -> {
                double p = w * iv.v2;
                scoresMap.addTo(iv.v1, p);
            });
        });
        

        //Negative neighborhood
        negativeNeighborhood.getNeighbors(uidx).forEach(vs -> {
            double w = pow(vs.v2, q);
            data.getUidxPreferences(vs.v1).forEach(iv -> {
                double p = w * iv.v2 * lambdaSim;
                if (negative) {
                	scoresMap.addTo(iv.v1, -p);
                } else {
                	scoresMap.addTo(iv.v1, p);
                }
            });
        });

        
        
        return scoresMap;
	}

}
