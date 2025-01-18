package es.uam.eps.knndissimilarities.utils;

import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.ranksys.core.util.tuples.Tuple2id;
import org.ranksys.core.util.tuples.Tuple2io;


import es.uam.eps.ir.ranksys.nn.neighborhood.Neighborhood;
import es.uam.eps.ir.ranksys.nn.sim.Similarity;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import it.unimi.dsi.fastutil.ints.IntArrayList;

public class RestrictedCombinedCachedNeighborhood implements Neighborhood
{

    private final IntArrayList[] idxla;
    private final DoubleArrayList[] simla;

    /**
     * Constructor that calculates and caches neighborhoods.
     *
     * @param n number of users/items
     * @param neighborhood generic neighborhood to be cached
     * @param sim the alternative similarity to use
     */
    public RestrictedCombinedCachedNeighborhood(int n, Neighborhood neighborhood, Similarity sim, boolean inverse)
    {

        this.idxla = new IntArrayList[n];
        this.simla = new DoubleArrayList[n];

        IntStream.range(0, n).parallel().forEach(idx -> {
            IntArrayList idxl = new IntArrayList();
            DoubleArrayList siml = new DoubleArrayList();
            // We keep the same order:
            neighborhood.getNeighbors(idx).forEach(is ->
            {
                idxl.add(is.v1);
                double antisim = sim.similarity(idx, is.v1);
                siml.add(is.v2 + (inverse ? -antisim : antisim));
            });
            idxla[idx] = idxl;
            simla[idx] = siml;
        });
    }

    /**
     * Constructor that caches a stream of previously calculated neighborhoods.
     *
     * @param n number of users/items
     * @param neighborhoods stream of already calculated neighborhoods
     * @param sim the alternative similarity to use
     */
    public RestrictedCombinedCachedNeighborhood(int n, Stream<Tuple2io<Stream<Tuple2id>>> neighborhoods, Similarity sim, boolean inverse)
    {
        this.idxla = new IntArrayList[n];
        this.simla = new DoubleArrayList[n];

        neighborhoods.forEach(un -> {
            int idx = un.v1;
            IntArrayList idxl = new IntArrayList();
            DoubleArrayList siml = new DoubleArrayList();
            un.v2.forEach(is -> {
                idxl.add(is.v1);
                double antisim = sim.similarity(idx, is.v1);
                siml.add(is.v2 + (inverse ? -antisim : antisim));
            });
            idxla[idx] = idxl;
            simla[idx] = siml;
        });
    }

    /**
     * Returns the neighborhood of a user/index.
     *
     * @param idx user/index whose neighborhood is calculated
     * @return stream of user/item-similarity pairs.
     */
    @Override
    public Stream<Tuple2id> getNeighbors(int idx)
    {
        if (idx < 0)
        {
            return Stream.empty();
        }
        IntArrayList idxl = idxla[idx];
        DoubleArrayList siml = simla[idx];
        if (idxl == null || siml == null)
        {
            return Stream.empty();
        }
        return IntStream.range(0, idxl.size()).mapToObj(i -> new Tuple2id(idxl.getInt(i), siml.getDouble(i)));
    }

}
