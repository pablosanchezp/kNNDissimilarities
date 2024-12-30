package es.uam.eps.knndissimilarities.sims;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity;

public class AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccardUser<U> extends UserSimilarity<U> {

	public AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccardUser(FastPreferenceData<U, ?> data, double threshold) {
        super(data, new AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard(data, threshold));
	}
}
