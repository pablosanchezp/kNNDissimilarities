package es.uam.eps.knndissimilarities.sims;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.TransposedPreferenceData;
import es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity;
import es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity;

public class AntiPositiveNegativeBinarySetItemsPerUserSimilarityItem<I> extends ItemSimilarity<I> {

	public AntiPositiveNegativeBinarySetItemsPerUserSimilarityItem(FastPreferenceData<?, I> data, double threshold) {
        super(data, new AntiPositiveNegativeBinarySetItemsPerUserSimilarity(new TransposedPreferenceData<>(data), threshold));
	}
}
