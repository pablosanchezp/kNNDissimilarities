package es.uam.eps.knndissimilarities.sims;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.TransposedPreferenceData;
import es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity;
import es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity;

public class AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccardItem<I> extends ItemSimilarity<I> {

	public AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccardItem(FastPreferenceData<?, I> data, double threshold) {
        super(data, new AntiPositiveNegativeBinarySetItemsPerUserSimilarityJaccard(new TransposedPreferenceData<>(data), threshold));
	}
}
