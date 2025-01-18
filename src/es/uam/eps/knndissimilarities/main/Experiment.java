package es.uam.eps.knndissimilarities.main;
import static org.ranksys.formats.parsing.Parsers.lp;
import static org.ranksys.formats.parsing.Parsers.sp;

import java.io.File;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.jooq.lambda.tuple.Tuple4;
import org.ranksys.formats.preference.SimpleRatingPreferencesReader;
import org.ranksys.formats.rec.RecommendationFormat;
import org.ranksys.formats.rec.SimpleRecommendationFormat;

import es.uam.eps.ir.antimetrics.antirel.BinaryAntiRelevanceModel;
import es.uam.eps.ir.attrrec.datamodel.feature.SimpleUserFeatureData;
import es.uam.eps.ir.attrrec.datamodel.feature.SimpleUserFeaturesReader;
import es.uam.eps.ir.attrrec.datamodel.feature.UserFeatureData;
import es.uam.eps.ir.attrrec.main.AttributeRecommendationUtils;
import es.uam.eps.ir.attrrec.metrics.recommendation.averages.WeightedAverageRecommendationMetricIgnoreNoRelevantUsersAndNaNs;
import es.uam.eps.ir.attrrec.metrics.recommendation.averages.WeightedModelUser;
import es.uam.eps.ir.attrrec.metrics.recommendation.averages.WeightedModelUser.UserMetricWeight;

import es.uam.eps.ir.attrrec.utils.PredicatesStrategies;
import es.uam.eps.ir.crossdomainPOI.metrics.system.RealAggregateDiversity;
import es.uam.eps.ir.crossdomainPOI.metrics.system.UserCoverage;
import es.uam.eps.ir.ranksys.core.preference.ConcatPreferenceData;
import es.uam.eps.ir.ranksys.core.preference.PreferenceData;
import es.uam.eps.ir.ranksys.core.preference.SimplePreferenceData;
import es.uam.eps.ir.ranksys.diversity.sales.metrics.AggregateDiversityMetric;
import es.uam.eps.ir.ranksys.diversity.sales.metrics.GiniIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.metrics.RecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.SystemMetric;
import es.uam.eps.ir.ranksys.metrics.rank.NoDiscountModel;
import es.uam.eps.ir.ranksys.metrics.rel.BinaryRelevanceModel;
import es.uam.eps.ir.ranksys.metrics.rel.NoRelevanceModel;
import es.uam.eps.ir.ranksys.metrics.rel.RelevanceModel;

import es.uam.eps.ir.ranksys.rec.Recommender;
import es.uam.eps.ir.seqawareev.main.ExperimentUtils;
import es.uam.eps.knndissimilarities.utils.AntiNeighborsUtils;
import es.uam.eps.knndissimilarities.utils.JsonProcessData;
import es.uam.eps.knndissimilarities.utils.ProcessData;


public class Experiment {

	// All options of the arguments
	// Option of the case
	private static final String OPT_CASE = "option";

	// Train and test files
	private static final String OPT_TRAIN_FILE = "trainFile";
	private static final String OPT_TEST_FILE = "testFile";
	private static final String OPT_MIN_ITEMS = "minRatingItems";
	private static final String OPT_MIN_USERS = "minRatingUsers";

	private static final String OPT_MIN_UNIQUE_RATINGS_ITEMS = "minUniqueRatingItems";
	private static final String OPT_MIN_UNIQUE_RATINGS_USERS = "minUniqueRatingUsers";

	// Number of items to recommend in the recommenders
	private static final String OPT_ITEMS_RECOMMENDED = "items recommended";

	// RankSys use the complete indexes of the files or not (users of both training
	// and test or not)
	private static final String OPT_COMPLETE_INDEXES = "complete indexes";

	// Feature variables
	private static final String OPT_USER_FEATURE_FILE = "user feature file";
	private static final String OPT_USER_FEATURE_SELECTED = "user feature selected";
	private static final String OPT_COMPUTE_USER_FILTER = "compute user filter";


	private static final String OPT_THRESHOLD = "relevance or matching threshold";

	// Output files
	private static final String OPT_OUT_RESULT_FILE = "output result file";

	// RankSys similarities and recommender
	private static final String OPT_RANKSYS_SIM = "ranksys similarity";
	private static final String OPT_RANKSYS_SIM2 = "ranksys similarity 2";

	private static final String OPT_RANKSYS_REC = "ranksys recommender";

	// Overwrite output file or not. Default value is false.
	private static final String OPT_OVERWRITE = "overwrite result";

	// Variables only used in rankSysEvaluation
	private static final String OPT_RECOMMENDED_FILE = "recommended file";
	private static final String OPT_CUTOFF = "cutoff";

	// Working with tourists and locals and bots
	private static final String OPT_MIN_DIFF_TIME = "min diff time";
	private static final String OPT_MAX_DIFF_TIME = "Max difference between timestamps";
	private static final String OPT_MIN_CLOSE_PREF_BOT = "min close pref bot";
	private static final String OPT_CONSECUTIVE_SAME_CHECKINS = "remove consecutive checkins if they items are the same";
	private static final String OPT_MAX_SPEED_CHECKINS = "max speed between checkisn for not removing";
	private static final String OPT_CITY = "city selection";

	// Knn recommenders
	private static final String OPT_NEIGH = "neighbours";

	// Parameters for RankGeoFM
	private static final String OPT_EPSILON = "epsilon";
	private static final String OPT_C = "c";

	// Parameters for Ranksys Recommender MatrixFactorization
	private static final String OPT_K = "k factorizer value";
	private static final String OPT_ALPHA = "alpha factorizer value";
	private static final String OPT_LAMBDA = "lambda factorizer value";
	private static final String OPT_NUM_INTERACTIONS = "num interactions factorizer value";

	// Parameters for ranksys non accuracy evaluation
	private static final String OPT_RANKSYS_RELEVANCE_MODEL = "ranksys relevance model";
	private static final String DEFAULT_COMPUTE_USER_FILTER = "false";
	private static final String OPT_RANKSYS_DISCOUNT_MODEL = "ranksys discount model";
	private static final String OPT_RANKSYS_BACKGROUND = "ranksys background for relevance model";
	private static final String OPT_RANKSYS_BASE = "ranksys base for discount model";
	private static final String OPT_MIN_DIFF_TIME_2 = "min diff time2";
	
	private static final String OPT_ANTI_RELEVANCE_THRESHOLD = "Anti relevance threshold";
	private static final String OPT_COMPUTE_ONLY_ACC = "compute also dividing by the number of users in the test set";
	private static final String OPT_COMPUTE_ANTI_METRICS = "Compute anti-metrics";

	// Mappings
	private static final String OPT_MAPPING_ITEMS = "file of mapping items";
	private static final String OPT_MAPPING_USERS = "file of mapping users";
	private static final String OPT_MAPPING_CATEGORIES = "file of mapping categories";

	private static final String OPT_NEW_DATASET = "file for the new dataset";
	private static final String OPT_COORD_FILE = "poi coordinate file";
	private static final String OPT_POI_CITY_FILE = "poi city mapping file";
	private static final String OPT_CITY_SIM_FILE = "city similarity file";
	private static final String OPT_WRAPPER_STRATEGY = "wrapper strategy for ranksys (removing timestamps)";
	private static final String OPT_WRAPPER_STRATEGY_TIMESTAMPS = "wrapper strategy for ranksys (strategy for the timestamps)";
	private static final String OPT_MATCHING_CATEGORY = "matching category for recommendation";

	// Parameters
	private static final String OPT_MAX_DISTANCE = "maximum distance";
	private static final String OPT_USE_SIGMOID = "useSigmoid";
	private static final String OPT_THETA_FACTORIZER = "theta factorizer value";
	private static final String OPT_ALPHA_FACTORIZER2 = "alpha factorizer (2) value";
	private static final String OPT_SVD_BETA = "Beta value for SVD";
	private static final String OPT_USING_WRAPPER = "using wrapper in ranksys recommenders";

	private static final String OPT_RECOMMENDATION_STRATEGY = "recommendation strategy";

	private static final String OPT_MODE = "specific mode of an algorithm";

	private static final String OPT_USER_MODEL_WGT = "user weight model weight";
	private static final String OPT_SCORE_FREQ = "use simple score or the frequency for the AvgDis";
	
	private static final String OPT_NORMALIZE = "apply normalization";
	private static final String OPT_INVERSE = "inverse";
	private static final String OPT_JACCARD = "applyJaccard";
	private static final String OPT_NEGANTI = "negative anti";

	
	// Parameters of Aggregation library
	private static final String OPT_NORM_AGGREGATE_LIBRARY = "norm of aggregate library";
	private static final String OPT_COMB_AGGREGATE_LIBRARY = "comb of aggregate library";
	private static final String OPT_WEIGHT_AGGREGATE_LIBRARY = "weight of aggregate library";

	// Some default values for recommenders as strings
	// Some default values from MFs
	private static final String DEFAULT_FACTORS = "20";
	private static final String DEFAULT_ITERACTIONS = "20";

	private static final String DEFAULT_ALPHA = "1";
	private static final String DEFAULT_LAMBDA = "0.1";

	// For evaluation
	private static final String DEFAULT_ITEMS_RECOMMENDED = "100";
	private static final String DEFAULT_RECOMMENDATION_STRATEGY = "TRAIN_ITEMS";
	private static final String DEFAULT_OVERWRITE = "false";
	private static final String DEFAULT_COMPUTE_ANTI_METRICS = "false";
	
	private static final String DEFAULT_SVD_LEARNING_RATE = "0.01f";
	private static final String DEFAULT_SVD_MAX_LEARNING_RATE = "1000f";
	private static final String DEFAULT_SVD_REG_USER = "0.01f";
	private static final String DEFAULT_SVD_REG_ITEM = "0.01f";
	private static final String DEFAULT_SVD_REG_BIAS = "0.01f";

	// Other default values for recommenders
	private static final String DEFAULT_COMPLETE_INDEXES = "false";
	private static final String DEFAULT_MAX_DISTANCE = "0.1";
	private static final String DEFAULT_ALPHA_HKV = "1";
	private static final String DEFAULT_SVDBETA = "1";
	private static final String DEFAULT_SCORE = "SIMPLE";
	private static final String DEFAULT_REG1 = "0.01"; // from mymedialite
	private static final String DEFAULT_REG2 = "0.001"; // from mymedialite

	private static final String DEFAULT_USER_MODEL_WGT = "NORMAL";
	private static final String DEFAULT_OPT_COMPUTE_ONLY_ACC = "true";
	
	
	// Some values for rank geoFM from Librec
	private static final String DEFAULT_EPSILON = "0.3";

	
	private static final String OPT_FEATURES_ITEM_COL = "features item column";
	
	public static void main(String[] args) throws Exception {

		String step = "";
		CommandLine cl = getCommandLine(args);
		if (cl == null) {
			System.out.println("Error in arguments");
			return;
		}

		// Obtain the arguments these 2 are obligatory
		step = cl.getOptionValue(OPT_CASE);
		String trainFile = cl.getOptionValue(OPT_TRAIN_FILE);

		System.out.println(step);
		switch (step) {
		
			case "DatasetTransform": {
				boolean removeDuplicates = false;
				System.out.println("-o DatasetTransform -trf completeFile dstTransformation charSplit removeduplicates?");
				System.out.println(Arrays.toString(args));
				if (args.length >= 7) {
					removeDuplicates = Boolean.parseBoolean(args[6]);
				}
				ProcessData.DatasetTransformation(trainFile, args[4], args[5],removeDuplicates);
			}
			break;
			
			case "Kcore":
			case "DatasetReduction": {
				System.out.println(Arrays.toString(args));
				String outputFile = cl.getOptionValue(OPT_OUT_RESULT_FILE);
				Integer minUserRatings = Integer.parseInt(cl.getOptionValue(OPT_MIN_USERS));
				Integer minItemRatings = Integer.parseInt(cl.getOptionValue(OPT_MIN_ITEMS));
				
				Integer minUserUniqueRatings = Integer.parseInt(cl.getOptionValue(OPT_MIN_UNIQUE_RATINGS_USERS, "0"));
				Integer minItemUniqueRatings = Integer.parseInt(cl.getOptionValue(OPT_MIN_UNIQUE_RATINGS_ITEMS, "0"));
				
				ProcessData.DatasetReductionRatings(trainFile, outputFile, minUserRatings, minItemRatings, minUserUniqueRatings, minItemUniqueRatings);
			}
				break;
		
				//Global split (temporal)
			case "TemporalGlobalSplit": {
				System.out.println("-o TemporalGlobalSplit -trf completeFile dstPathTrain dstPathTest trainPercent");
				System.out.println(Arrays.toString(args));
				ProcessData.DatasetTemporalGlobalSplit(trainFile, args[4], args[5], Double.parseDouble(args[6]));
			}
				break;
			
			case "processJSONGoodReads":{
				String orf = cl.getOptionValue(OPT_OUT_RESULT_FILE);
				JsonProcessData.processJSONGoodReads(trainFile, orf);
			}
			break;
			
			// Method to transform implicit to explicit ratings
			case "ImplicitToExplicit": {
				System.out.println("-o ImplicitToExplicit -trf originalImplicit explicitTransform charSplit");
				System.out.println(Arrays.toString(args));
				ProcessData.implicitToExplicitMaxRatings(trainFile, args[4], args[5]);
			}
				break;
			
			case "lastFmGroupByColumn": {
				System.out.println(Arrays.toString(args));
				String outputFile = cl.getOptionValue(OPT_OUT_RESULT_FILE);
				Integer column = Integer.parseInt(cl.getOptionValue(OPT_FEATURES_ITEM_COL));
				ProcessData.lastFmGroupByColumn(trainFile, column, outputFile);
			}
			break;
				
			case "simpleRandomSplit": {
				System.out.println(Arrays.toString(args));
				
				long seed = System.currentTimeMillis();
				if (args.length >= 8) {
					System.out.println("Using custom seed");
					seed = Long.parseLong(args[7]);
				} 

				ProcessData.randomSplit(trainFile, args[4], args[5], Double.parseDouble(args[6]), seed);
			}
				break;
				
			case "ranksysOnlyComplete": {
				System.out.println(Arrays.toString(args));

				String outputFile = cl.getOptionValue(OPT_OUT_RESULT_FILE);
				String ranksysSimilarity = cl.getOptionValue(OPT_RANKSYS_SIM);
				String ranksysSimilarity2 = cl.getOptionValue(OPT_RANKSYS_SIM2);


				String ranksysRecommender = cl.getOptionValue(OPT_RANKSYS_REC);
				String testFile = cl.getOptionValue(OPT_TEST_FILE);

				// MFs Parameters
				Integer numIterations = Integer.parseInt(cl.getOptionValue(OPT_NUM_INTERACTIONS, DEFAULT_ITERACTIONS));
				Integer numFactors = Integer.parseInt(cl.getOptionValue(OPT_K, DEFAULT_FACTORS));
				


				Double threshold = Double.parseDouble(cl.getOptionValue(OPT_THRESHOLD, "0"));
				boolean inverse = Boolean.parseBoolean(cl.getOptionValue(OPT_INVERSE, "true")); 
				boolean applyJaccard = Boolean.parseBoolean(cl.getOptionValue(OPT_JACCARD, "true")); 
				
				String combiner = cl.getOptionValue(OPT_COMB_AGGREGATE_LIBRARY);
				String normalizer = cl.getOptionValue(OPT_NORM_AGGREGATE_LIBRARY);

				Boolean negativeAnti = Boolean.parseBoolean(cl.getOptionValue(OPT_NEGANTI, "true"));

				
				// Rank Geo
			
				// HKV Factorizer
				Double alphaFactorizer = Double.parseDouble(cl.getOptionValue(OPT_ALPHA, DEFAULT_ALPHA_HKV));
				Double lambdaFactorizer = Double.parseDouble(cl.getOptionValue(OPT_LAMBDA, DEFAULT_LAMBDA));




				final int numberItemsRecommend = Integer.parseInt(cl.getOptionValue(OPT_ITEMS_RECOMMENDED));
				int neighbours = Integer.parseInt(cl.getOptionValue(OPT_NEIGH));

				String recommendationStrategy = cl.getOptionValue(OPT_RECOMMENDATION_STRATEGY,
						DEFAULT_RECOMMENDATION_STRATEGY);



				if (ranksysSimilarity == null) {
					System.out.println("Not working with any recommender using similarities");
				}

				String overwrite = cl.getOptionValue(OPT_OVERWRITE, DEFAULT_OVERWRITE);

				File f = new File(outputFile);
				if (f.exists() && !f.isDirectory() && !Boolean.parseBoolean(overwrite)) {
					System.out.println("Ignoring " + f + " because it already exists");
					return;
				}

				boolean completeOrNot = Boolean
						.parseBoolean(cl.getOptionValue(OPT_COMPLETE_INDEXES, DEFAULT_COMPLETE_INDEXES));

				FastPreferenceData<Long, Long> trainPrefData = null;
				FastPreferenceData<Long, Long> testPrefData = null;

				trainPrefData = ExperimentUtils.loadTrainFastPreferenceData(trainFile, testFile, completeOrNot, true);

				

				System.out.println("Not using wrapper (working with no timestamps)");

				Recommender<Long, Long> rankSysrec = AntiNeighborsUtils.obtRankSysRecommeder(ranksysRecommender, ranksysSimilarity, ranksysSimilarity2,
								trainPrefData, neighbours, numFactors, alphaFactorizer, lambdaFactorizer, numIterations, inverse, applyJaccard, threshold, combiner, normalizer, lambdaFactorizer, negativeAnti);

				if (rankSysrec == null) {
					System.out.println("RECOMMENDER IS NULL");
				}
				System.out.println("Analyzing " + testFile);
				System.out.println("Recommender file " + outputFile);

				testPrefData = ExperimentUtils.loadTrainFastPreferenceData(trainFile, testFile, true, false);

				System.out.println(
						"Writing recommended file. Not items candidates file provided. All candidates are the items not seen by that user in train.");
				PredicatesStrategies.ranksysWriteRanking(trainPrefData, testPrefData, rankSysrec, outputFile,
						numberItemsRecommend, AttributeRecommendationUtils.obtRecommendationStrategy(recommendationStrategy));

			}
				break;
		
				
			case "ParseMyMediaLite": {
					System.out.println("-o ParseMyMediaLite -trf myMediaLiteRecommendation testFile newRecommendation");
					System.out.println(Arrays.toString(args));
					Tuple4<List<Long>, List<Long>, List<Long>, List<Long>> indexes = ExperimentUtils
							.retrieveTrainTestIndexes(args[4], args[4], false, lp, lp);

					List<Long> usersTest = indexes.v1;
					FastUserIndex<Long> userIndexTest = SimpleFastUserIndex.load(usersTest.stream());

					ProcessData.parseMyMediaLite(trainFile, userIndexTest, args[5]);
				}
					break;	
				
		
		// Ranksys with non accuracy metrics
				case "ranksysNonAccuracyWithoutFeatureMetricsEvaluation":
				case "ranksysNonAccuracyMetricsEvaluation":
				case "ranksysNonAccuracyMetricsEvaluationPerUser":
				case "ranksysNonAccuracyWithoutFeatureMetricsEvaluationPerUser": {
					/*
					 * -Train file -Test file -Recommended file -Item feature file -Ranksys Metric
					 * -Output file -Threshold -Cutoff
					 */
					System.out.println(Arrays.toString(args));

					String recommendedFile = cl.getOptionValue(OPT_RECOMMENDED_FILE);
					String userFeaturFile = cl.getOptionValue(OPT_USER_FEATURE_FILE);
					String userSelFeature = cl.getOptionValue(OPT_USER_FEATURE_SELECTED);

					int threshold = Integer.parseInt(cl.getOptionValue(OPT_THRESHOLD));

					
					String cutoffs = cl.getOptionValue(OPT_CUTOFF);
					String overwrite = cl.getOptionValue(OPT_OVERWRITE, DEFAULT_OVERWRITE);
					String userWeightModel = cl.getOptionValue(OPT_USER_MODEL_WGT, DEFAULT_USER_MODEL_WGT);


					Boolean computeOnlyAcc = Boolean
							.parseBoolean(cl.getOptionValue(OPT_COMPUTE_ONLY_ACC, DEFAULT_OPT_COMPUTE_ONLY_ACC));

					Boolean computeUserFilter = Boolean
							.parseBoolean(cl.getOptionValue(OPT_COMPUTE_USER_FILTER, DEFAULT_COMPUTE_USER_FILTER));
					



					Boolean isPerUser = step.contains("PerUser");

					// This ones can be more than one file
					String outputFileS = cl.getOptionValue(OPT_OUT_RESULT_FILE);
					String testFileS = cl.getOptionValue(OPT_TEST_FILE);

					//output and test files need to be separated by -
					String outputFileArr[] = outputFileS.split(",");
					String testFileArr[] = testFileS.split(",");
					
					if (outputFileArr.length == 1) {
						File f = new File(outputFileArr[0]);
						
						// If file of ranksys evaluation already exist then nothing
						if (f.exists() && !f.isDirectory() && Boolean.parseBoolean(overwrite) == false) {
							System.out.println("Ignoring " + f + " because it already exists");
							break;
						}
					}



					if (isPerUser) {
						System.out.println("Per user evaluation. We will compute the metrics for every user indvidually.");
					} else {
						System.out.println("Normal evaluation, computing the average of the results in the metrics.");
					}



					final PreferenceData<Long, Long> trainData = SimplePreferenceData
							.load(SimpleRatingPreferencesReader.get().read(trainFile, lp, lp));

					
					final PreferenceData<Long, Long> originalRecommendedData = SimplePreferenceData
							.load(SimpleRatingPreferencesReader.get().read(recommendedFile, lp, lp));
					



					

					
					for (int i = 0; i < outputFileArr.length; i++) {
						String outputFile = outputFileArr[i];
						String testFile = testFileArr[i];
						
						File f = new File(outputFile);
						
						// If file of ranksys evaluation already exist then nothing
						if (f.exists() && !f.isDirectory() && Boolean.parseBoolean(overwrite) == false) {
							System.out.println("Ignoring " + f + " because it already exists");
							continue;
						}


						// End of test Temporal data

						
						
						final PreferenceData<Long, Long> testDataNoFilter = SimplePreferenceData
								.load(SimpleRatingPreferencesReader.get().read(testFile, lp, lp));
						
						final PreferenceData<Long, Long> testData = testDataNoFilter;
						
						final PreferenceData<Long, Long> totalData = new ConcatPreferenceData<>(trainData, testData);

						final Set<Long> testUsers = testData.getUsersWithPreferences().collect(Collectors.toSet());
						
						// recommended data has to be filtered to avoid evaluating users not in test
						final PreferenceData<Long, Long> recommendedData = AntiNeighborsUtils
								.filterPreferenceData(originalRecommendedData, testUsers, null);

						////////////////////////
						// INDIVIDUAL METRICS //
						////////////////////////

						// Binary relevance and anti relevance model for ranking metrics
						BinaryRelevanceModel<Long, Long> binRel = new BinaryRelevanceModel<>(false, testData, threshold);
						
						// Relevance model for novelty/diversity (can be with or without relevance)
						RelevanceModel<Long, Long> selectedRelevance = new NoRelevanceModel<>();

						

						int numUsersTest = testData.numUsersWithPreferences();
						int numUsersRecommended = recommendedData.numUsersWithPreferences();
						int numItems = totalData.numItemsWithPreferences(); // Num items with preferences in the data

						
						System.out.println("\n\nNum users in the test set " + numUsersTest);
						System.out.println("\n\nNum users to whom we have made recommendations " + numUsersRecommended);
						System.out.println("Modified ratios normalization");

						Map<String, SystemMetric<Long, Long>> sysMetrics = new HashMap<>();
						// Ranking metrics (avg for recommendation)
						Map<String, RecommendationMetric<Long, Long>> recMetricsAvgRelUsers = new HashMap<>();
						Map<String, RecommendationMetric<Long, Long>> recMetricsAllRecUsers = new HashMap<>();

						String[] differentCutoffs = cutoffs.split(",");

						

						
						UserFeatureData<Long, String, Double> ufD = null;
						if (!isPerUser) {
							if (userFeaturFile != null) {
								ufD = SimpleUserFeatureData.load(SimpleUserFeaturesReader.get().read(userFeaturFile, lp, sp));
							}
						}
						
						
						for (String cutoffS : differentCutoffs) {
							int cutoff = Integer.parseInt(cutoffS);
							AntiNeighborsUtils.addMetrics(recMetricsAvgRelUsers, recMetricsAllRecUsers,
									threshold, cutoff, trainData, testData, selectedRelevance, binRel, new NoDiscountModel(), computeOnlyAcc);

		
								

								sysMetrics.put("aggrdiv@" + cutoff, new AggregateDiversityMetric<>(cutoff, selectedRelevance));
								sysMetrics.put("gini@" + cutoff, new GiniIndex<>(cutoff, numItems));
								sysMetrics.put("RealAD@" + cutoff, new RealAggregateDiversity<Long, Long>(cutoff));

						}
						
						//User cov goes without cutoff
						sysMetrics.put("usercov", new UserCoverage<Long, Long>());

						// Average of all only for normal evaluation, not per user
						if (!isPerUser) {

							UserMetricWeight weight = AntiNeighborsUtils.obtUserMetricWeight(userWeightModel);
							System.out.println("Working with user weight " + weight.toString());

							WeightedModelUser<Long, Long, String, Double> wmu = new WeightedModelUser<>(trainData, weight,
									userSelFeature, ufD);

							recMetricsAvgRelUsers.forEach((name, metric) -> sysMetrics.put(name + "_rec",
									new WeightedAverageRecommendationMetricIgnoreNoRelevantUsersAndNaNs<>(metric, binRel,
											wmu)));

							recMetricsAllRecUsers.forEach((name, metric) -> sysMetrics.put(name + "_rec",
									new WeightedAverageRecommendationMetricIgnoreNoRelevantUsersAndNaNs<>(metric, binRel,
											wmu)));


							RecommendationFormat<Long, Long> format = new SimpleRecommendationFormat<>(lp, lp);

							/*
							 * format.getReader(recommendedFile).readAll() .forEach(rec ->
							 * sysMetrics.values().forEach(metric -> metric.add(rec)));
							 */

							System.out.println("Computing user filter: " + computeUserFilter);
							

							format.getReader(recommendedFile).readAll().forEach(rec -> sysMetrics.values().forEach(metric -> metric.add(rec)));

							PrintStream out = new PrintStream(new File(outputFile));
							sysMetrics.forEach((name, metric) -> out.println(name + "\t" + metric.evaluate()));
							out.close();
						}


					}

				}
		}

	}

	private static CommandLine getCommandLine(String[] args) {
		Options options = new Options();

		// Number of the case
		Option caseIdentifier = new Option("o", OPT_CASE, true, "option of the case");
		caseIdentifier.setRequired(true);
		options.addOption(caseIdentifier);

		// Train file
		Option trainFile = new Option("trf", OPT_TRAIN_FILE, true, "input file train path");
		trainFile.setRequired(true);
		options.addOption(trainFile);

		// Here not required
		// TestFile file
		Option testFile = new Option("tsf", OPT_TEST_FILE, true, "input file test path");
		testFile.setRequired(false);
		options.addOption(testFile);

		// Neighbours
		Option neighbours = new Option("n", OPT_NEIGH, true, "neighbours");
		neighbours.setRequired(false);
		options.addOption(neighbours);

		// NumberItemsRecommended
		Option numberItemsRecommended = new Option("nI", OPT_ITEMS_RECOMMENDED, true, "Number of items recommended");
		numberItemsRecommended.setRequired(false);
		options.addOption(numberItemsRecommended);

		// OutResultfile
		Option outfile = new Option("orf", OPT_OUT_RESULT_FILE, true, "output result file");
		outfile.setRequired(false);
		options.addOption(outfile);

		// Ranksys similarity
		Option rankSysSim = new Option("rs", OPT_RANKSYS_SIM, true, "ranksys similarity");
		rankSysSim.setRequired(false);
		options.addOption(rankSysSim);
		
		// Ranksys similarity
		Option rankSysSim2 = new Option("rs2", OPT_RANKSYS_SIM2, true, "ranksys similarity");
		rankSysSim2.setRequired(false);
		options.addOption(rankSysSim2);		

		// Ranksys recommender
		Option rankSysRecommender = new Option("rr", OPT_RANKSYS_REC, true, "ranksys recommeder");
		rankSysRecommender.setRequired(false);
		options.addOption(rankSysRecommender);

		// Overwrite result
		Option outputOverwrite = new Option("ovw", OPT_OVERWRITE, true, "overwrite");
		outputOverwrite.setRequired(false);
		options.addOption(outputOverwrite);

		// Feature reading file
		Option recommendedFile = new Option("rf", OPT_RECOMMENDED_FILE, true, "recommended file");
		recommendedFile.setRequired(false);
		options.addOption(recommendedFile);

		// RankSysMetric
		Option ranksysCutoff = new Option("rc", OPT_CUTOFF, true, "ranksyscutoff");
		ranksysCutoff.setRequired(false);
		options.addOption(ranksysCutoff);

		// RankSys factorizers (k)
		Option kFactorizer = new Option("kFactorizer", OPT_K, true, "k factorizer");
		kFactorizer.setRequired(false);
		options.addOption(kFactorizer);

		// RankSys factorizers (alpha)
		Option alhpaFactorizer = new Option("aFactorizer", OPT_ALPHA, true, "alpha factorizer");
		alhpaFactorizer.setRequired(false);
		options.addOption(alhpaFactorizer);

		// RankSys factorizers (lambda)
		Option lambdaFactorizer = new Option("lFactorizer", OPT_LAMBDA, true, "lambda factorizer");
		lambdaFactorizer.setRequired(false);
		options.addOption(lambdaFactorizer);

		// RankSys factorizers (numInteractions)
		Option numInteractionsFact = new Option("nIFactorizer", OPT_NUM_INTERACTIONS, true,
				"numInteractions factorizer");
		numInteractionsFact.setRequired(false);
		options.addOption(numInteractionsFact);

		// Factorizers (theta)
		Option thetaFactorizer = new Option("thetaFactorizer", OPT_THETA_FACTORIZER, true, "theta factorizer");
		thetaFactorizer.setRequired(false);
		options.addOption(thetaFactorizer);

		// factorizers (alpha2)
		Option alphaFactorizer2 = new Option("aFactorizer2", OPT_ALPHA_FACTORIZER2, true, "alpha factorizer (2)");
		alphaFactorizer2.setRequired(false);
		options.addOption(alphaFactorizer2);

		// RansksyLibrary NonAccuracyEvaluationParameters
		Option ranksysRelevanceModel = new Option("ranksysRelModel", OPT_RANKSYS_RELEVANCE_MODEL, true,
				"ranksys relevance model");
		ranksysRelevanceModel.setRequired(false);
		options.addOption(ranksysRelevanceModel);

		Option ranksysDiscountModel = new Option("ranksysDiscModel", OPT_RANKSYS_DISCOUNT_MODEL, true,
				"ranksys discount model");
		ranksysDiscountModel.setRequired(false);
		options.addOption(ranksysDiscountModel);

		Option ranksysBackground = new Option("ranksysBackRel", OPT_RANKSYS_BACKGROUND, true,
				"ranksys background for relevance model");
		ranksysBackground.setRequired(false);
		options.addOption(ranksysBackground);

		Option ranksysBase = new Option("ranksysBaseDisc", OPT_RANKSYS_BASE, true, "ranksys base for discount model");
		ranksysBase.setRequired(false);
		options.addOption(ranksysBase);

		// File containing 2 columns, old id and new id, for datasets that we have
		// changed the ids (for items)
		Option itemsMapping = new Option("IMapping", OPT_MAPPING_ITEMS, true, "mapping file of items");
		itemsMapping.setRequired(false);
		options.addOption(itemsMapping);

		// File containing 2 columns, old id and new id, for datasets that we have
		// changed the ids
		Option usersMapping = new Option("UMapping", OPT_MAPPING_USERS, true, "mapping file of users");
		usersMapping.setRequired(false);
		options.addOption(usersMapping);

		// File containing 2 columns, old id and new id, for datasets that we have
		// changed the ids
		Option mapCategories = new Option("CMapping", OPT_MAPPING_CATEGORIES, true, "mapping file of categories");
		mapCategories.setRequired(false);
		options.addOption(mapCategories);

		// Option for creating a new dataset when transforming the users and items
		Option newDatasetFile = new Option("newDataset", OPT_NEW_DATASET, true, "new dataset destination");
		newDatasetFile.setRequired(false);
		options.addOption(newDatasetFile);

		// Option for creating a new dataset when transforming the users and items
		Option poiCoordFile = new Option("coordFile", OPT_COORD_FILE, true, "poi coordinates file");
		poiCoordFile.setRequired(false);
		options.addOption(poiCoordFile);

		Option poiCityFile = new Option("poiCityFile", OPT_POI_CITY_FILE, true, "poi city mapping file");
		poiCityFile.setRequired(false);
		options.addOption(poiCityFile);

		Option citySimFile = new Option("citySimFile", OPT_CITY_SIM_FILE, true, "city similarity file");
		citySimFile.setRequired(false);
		options.addOption(citySimFile);

		// Option of the wrapper strategy to parse datasets or to use it in ranksys
		Option wrapperStrategyPreferences = new Option("wStrat", OPT_WRAPPER_STRATEGY, true,
				"strategy to use in the wrapper");
		wrapperStrategyPreferences.setRequired(false);
		options.addOption(wrapperStrategyPreferences);

		// Option of the wrapper strategy to parse datasets or to use it in ranksys
		Option wrapperStrategyTimeStamps = new Option("wStratTime", OPT_WRAPPER_STRATEGY_TIMESTAMPS, true,
				"strategy to use in the wrapper (for timestamps)");
		wrapperStrategyTimeStamps.setRequired(false);
		options.addOption(wrapperStrategyTimeStamps);

		Option matchingCategory = new Option("matchCat", OPT_MATCHING_CATEGORY, true,
				"matching category for candidate items");
		matchingCategory.setRequired(false);
		options.addOption(matchingCategory);

		Option usingWrapper = new Option("usingWrapper", OPT_USING_WRAPPER, true, "using wrapper in ranksys");
		usingWrapper.setRequired(false);
		options.addOption(usingWrapper);

		Option useCompleteIndex = new Option("cIndex", OPT_COMPLETE_INDEXES, true,
				"use the complete indexes (train + test)");
		useCompleteIndex.setRequired(false);
		options.addOption(useCompleteIndex);

		Option recommendationStrategy = new Option("recStrat", OPT_RECOMMENDATION_STRATEGY, true,
				"recommendation strategy");
		recommendationStrategy.setRequired(false);
		options.addOption(recommendationStrategy);

		Option scoreFreq = new Option("scoreFreq", OPT_SCORE_FREQ, true, "simple or frequency");
		scoreFreq.setRequired(false);
		options.addOption(scoreFreq);

		Option maxDist = new Option("maxDist", OPT_MAX_DISTANCE, true, "maximum distance");
		maxDist.setRequired(false);
		options.addOption(maxDist);

		Option mode = new Option("mode", OPT_MODE, true, "mode of an algorithm");
		mode.setRequired(false);
		options.addOption(mode);

		Option useSigmoid = new Option("useSigmoid", OPT_USE_SIGMOID, true, "use sigmoid");
		useSigmoid.setRequired(false);
		options.addOption(useSigmoid);

		Option maxDiffBetweenTimestamps = new Option("maxDiffTime", OPT_MAX_DIFF_TIME, true, "maxDifftime");
		maxDiffBetweenTimestamps.setRequired(false);
		options.addOption(maxDiffBetweenTimestamps);

		Option minDiffBetweenTimestamps = new Option("minDiffTime", OPT_MIN_DIFF_TIME, true, "minDifftime");
		minDiffBetweenTimestamps.setRequired(false);
		options.addOption(minDiffBetweenTimestamps);

		Option minClosePrefBot = new Option("minClosePrefBot", OPT_MIN_CLOSE_PREF_BOT, true, "minDifftime");
		minClosePrefBot.setRequired(false);
		options.addOption(minClosePrefBot);

		Option userFeatureFile = new Option("uff", OPT_USER_FEATURE_FILE, true, "user feature file");
		userFeatureFile.setRequired(false);
		options.addOption(userFeatureFile);

		Option userFeatureSelected = new Option("ufs", OPT_USER_FEATURE_SELECTED, true, "user feature selected");
		userFeatureSelected.setRequired(false);
		options.addOption(userFeatureSelected);

		Option computeUserFilter = new Option("compUF", OPT_COMPUTE_USER_FILTER, true, "compute user filter");
		computeUserFilter.setRequired(false);
		options.addOption(computeUserFilter);

		Option epsilon = new Option("epsilon", OPT_EPSILON, true, "epsilon");
		epsilon.setRequired(false);
		options.addOption(epsilon);

		Option userModelWeight = new Option("userModelW", OPT_USER_MODEL_WGT, true, "user model weight");
		userModelWeight.setRequired(false);
		options.addOption(userModelWeight);

		// threshold
		Option threshold = new Option("thr", OPT_THRESHOLD, true, "relevance or matching threshold");
		threshold.setRequired(false);
		options.addOption(threshold);

		Option consecutiveSameCheckins = new Option("consecutiveSameCheckins", OPT_CONSECUTIVE_SAME_CHECKINS, true,
				"consecutiveSameCheckins");
		consecutiveSameCheckins.setRequired(false);
		options.addOption(consecutiveSameCheckins);

		Option maxSpeed = new Option("maxSpeed", OPT_MAX_SPEED_CHECKINS, true, "maxSpeed");
		maxSpeed.setRequired(false);
		options.addOption(maxSpeed);

		Option minDiffBetweenTimestamps2 = new Option("minDiffTime2", OPT_MIN_DIFF_TIME_2, true, "minDifftime2");
		minDiffBetweenTimestamps2.setRequired(false);
		options.addOption(minDiffBetweenTimestamps2);

		// numberMinRatItems
		Option numberMinRatItems = new Option("mri", OPT_MIN_ITEMS, true, "Minimum number of item ratings");
		numberMinRatItems.setRequired(false);
		options.addOption(numberMinRatItems);
		
		// ItemColFeature
		Option itemColFeature = new Option("fic", OPT_FEATURES_ITEM_COL, true, "item column of feature file");
		itemColFeature.setRequired(false);
		options.addOption(itemColFeature);

		// numberMinRatUsers
		Option numberMinRatUsers = new Option("mru", OPT_MIN_USERS, true, "Minimum number of user ratings");
		numberMinRatUsers.setRequired(false);
		options.addOption(numberMinRatUsers);

		// numberMinRatItems
		Option numberMinUniqueRatItems = new Option("mUniqri", OPT_MIN_UNIQUE_RATINGS_ITEMS, true,
				"Minimum number of item unique ratings");
		numberMinUniqueRatItems.setRequired(false);
		options.addOption(numberMinUniqueRatItems);

		// numberMinRatUsers
		Option numberMinUniqueRatUsers = new Option("mUniqru", OPT_MIN_UNIQUE_RATINGS_USERS, true,
				"Minimum number of user unique ratings");
		numberMinUniqueRatUsers.setRequired(false);
		options.addOption(numberMinUniqueRatUsers);

		Option citySelected = new Option("city", OPT_CITY, true, "city selection");
		citySelected.setRequired(false);
		options.addOption(citySelected);
		
		Option inverse = new Option("inverse", OPT_INVERSE, true, "apply inverse");
		inverse.setRequired(false);
		options.addOption(inverse);
		
		Option normalize = new Option("normalize", OPT_NORMALIZE, true, "normalize");
		normalize.setRequired(false);
		options.addOption(normalize);
		
		Option applyJaccard = new Option("applyJaccard", OPT_JACCARD, true, "applyJaccard");
		applyJaccard.setRequired(false);
		options.addOption(applyJaccard);
		
		Option negAntiN = new Option("negAntiN", OPT_NEGANTI, true, "negative anti-neigh");
		negAntiN.setRequired(false);
		options.addOption(negAntiN);
		
		// RankLibrary norm value
		Option normAggregate = new Option("normAgLib", OPT_NORM_AGGREGATE_LIBRARY, true,
				"normalization aggregate library");
		normAggregate.setRequired(false);
		options.addOption(normAggregate);

		// RankLibrary comb value
		Option combAggregate = new Option("combAgLib", OPT_COMB_AGGREGATE_LIBRARY, true, "combination aggregate library");
		combAggregate.setRequired(false);
		options.addOption(combAggregate);

		// RankLibrary norm value
		Option weightAggregate = new Option("weightAgLib", OPT_WEIGHT_AGGREGATE_LIBRARY, true,
				"weight aggregate library");
		weightAggregate.setRequired(false);
		options.addOption(weightAggregate);
		
		Option antiMetrics = new Option("computeAntiMetrics", OPT_COMPUTE_ANTI_METRICS, true,
				"boolean in order to compute anti metrics");
		antiMetrics.setRequired(false);
		options.addOption(antiMetrics);

		Option antiRelevanceThreshold = new Option("antiRelTh", OPT_ANTI_RELEVANCE_THRESHOLD, true,
				"integer in order to indicate the antiRelevanceTh");
		antiRelevanceThreshold.setRequired(false);
		options.addOption(antiRelevanceThreshold);
		
		Option computeOnlyAccuracy = new Option("onlyAcc", OPT_COMPUTE_ONLY_ACC, true, "if we are computing just accuracy metrics or also novelty-diversity");
		computeOnlyAccuracy.setRequired(false);
		options.addOption(computeOnlyAccuracy);
		
		

		CommandLineParser parser = new DefaultParser();
		HelpFormatter formatter = new HelpFormatter();
		CommandLine cmd;

		try {
			cmd = parser.parse(options, args);
		} catch (ParseException e) {
			System.out.println(e.getMessage());
			formatter.printHelp("utility-name", options);

			return null;
		}
		return cmd;

	}

}
