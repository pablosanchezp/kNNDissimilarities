package es.uam.eps.knndissimilarities.utils;

import es.uam.eps.ir.ranksys.nn.item.neighborhood.ItemNeighborhood;
import es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity;

public class ItemRestrictedCombinedCachedNeighborhood<I> extends ItemNeighborhood<I>
{
    /**
     * Constructor
     * @param neighborhood the original neighborhood.
     * @param sim the item similarity
     * @param inverse true if we want to substract the value of the second similarity, false if we want to add both sims.
     */
    public ItemRestrictedCombinedCachedNeighborhood(int n, ItemNeighborhood<I> neighborhood, ItemSimilarity<I> sim, boolean inverse)
    {
        super(neighborhood, new RestrictedCombinedCachedNeighborhood(n, neighborhood, sim, inverse));
    }
}
