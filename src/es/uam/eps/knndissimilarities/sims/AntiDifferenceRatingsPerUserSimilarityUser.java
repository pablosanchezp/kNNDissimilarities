package es.uam.eps.knndissimilarities.sims;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity;

public class AntiDifferenceRatingsPerUserSimilarityUser<U> extends UserSimilarity<U> {

	public AntiDifferenceRatingsPerUserSimilarityUser(FastPreferenceData<U, ?> data, boolean applyJaccard) {
        super(data, new AntiDifferenceRatingsPerUserSimilarity(data, applyJaccard));
	}
}
