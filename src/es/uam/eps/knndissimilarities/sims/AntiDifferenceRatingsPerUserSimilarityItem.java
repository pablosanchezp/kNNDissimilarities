package es.uam.eps.knndissimilarities.sims;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.TransposedPreferenceData;
import es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity;

public class AntiDifferenceRatingsPerUserSimilarityItem<I> extends ItemSimilarity<I> {

	public AntiDifferenceRatingsPerUserSimilarityItem(FastPreferenceData<?, I> data, boolean applyJaccard) {
        super(data, new AntiDifferenceRatingsPerUserSimilarity(new TransposedPreferenceData<>(data), applyJaccard));
	}
}

