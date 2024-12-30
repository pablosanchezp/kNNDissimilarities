package es.uam.eps.knndissimilarities.sims;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity;

public class AntiPositiveNegativeBinarySetItemsPerUserSimilarityUser<U> extends UserSimilarity<U> {

	public AntiPositiveNegativeBinarySetItemsPerUserSimilarityUser(FastPreferenceData<U, ?> data, double threshold) {
        super(data, new AntiPositiveNegativeBinarySetItemsPerUserSimilarity(data, threshold));
	}
}
