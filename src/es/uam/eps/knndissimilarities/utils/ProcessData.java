package es.uam.eps.knndissimilarities.utils;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;


public final class ProcessData {
	
	/***
	 * Method to group the preferences of lastfm by a specific column. The lastfm dataset contains
	 * repeated listetings, so we can group all the data by an specific column to create a more suitable
	 * data for recommenders
	 * @param inputFile
	 * @param columnGroup
	 * @param outputFile
	 */
	public static void lastFmGroupByColumn(String inputFile, int columnGroup, String outputFile) {
		Map<String, Map<String, Integer>> mapping = new HashMap<>();
		
    	PrintStream outPutResult;
		try {
    		Stream<String> stream = Files.lines(Paths.get(inputFile));
    		stream.forEach(line -> {
    			String data [] = line.split("\t");
    			String user = data[0];
    			String groupingItem = data[columnGroup];
    			if (data.length != 6 || groupingItem.equals("") || groupingItem.equals(" ")) {
    				System.out.println(data);
    			}
    			
    			if (mapping.get(user) == null) {
    				mapping.put(user, new HashMap<>());
    			}
    			
    			
    			if (mapping.get(user).get(groupingItem) == null) {
    				mapping.get(user).put(groupingItem, 0);
    			}
    			int c = mapping.get(user).get(groupingItem);
				mapping.get(user).put(groupingItem, c + 1);
    		});
    		stream.close();
			outPutResult = new PrintStream(outputFile);
			for (String user: mapping.keySet()) {
				for (Map.Entry<String, Integer> entry : mapping.get(user).entrySet()) {
					outPutResult.println(user + "\t" + entry.getKey() + "\t" + entry.getValue());
				}
			}
	    	outPutResult.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void DatasetTransformation(String srcPath, String dstPath, String characterSplit,
			boolean removeDuplicates) {
		Map<String, Long> element1s = new TreeMap<String, Long>();
		Map<String, Long> element2s = new TreeMap<String, Long>();

		// Id user, id item rating and timestamp
		// The data to store is a map of users, with a map of items and a list
		// associated with the user and the item
		Map<Long, Map<Long, List<Tuple2<Double, Long>>>> storationUsers = new TreeMap<Long, Map<Long, List<Tuple2<Double, Long>>>>();

		// Variables to customize
		boolean ignoreFirstLine = false; // If we want to ignore first line, switch to true
		int columnFirstElement = 0;
		int columnSecondElement = 1;
		int columnRating = 2;
		int columnTimeStamp = 3;

		BufferedWriter writer = null;
		AtomicLong totalElements1 = new AtomicLong(1L);
		AtomicLong totalElements2 = new AtomicLong(1L);

		Stream<String> stream = null;
		try {
			if (ignoreFirstLine) {
				stream = Files.lines(Paths.get(srcPath)).skip(1); // skipping the first line
			} else {
				stream = Files.lines(Paths.get(srcPath));
			}

			stream.forEach(line -> {
				String[] data = line.split(characterSplit);
				// Transforming the new ids
				if (element1s.get(data[columnFirstElement]) == null) {
					element1s.put(data[columnFirstElement], totalElements1.get());
					totalElements1.incrementAndGet();
				}
				if (element2s.get(data[columnSecondElement]) == null) {
					element2s.put(data[columnSecondElement], totalElements2.get());
					totalElements2.incrementAndGet();
				}

				Long idUserTransformed = element1s.get(data[columnFirstElement]);
				Long idItemTransformed = element2s.get(data[columnSecondElement]);
				double rating = Double.parseDouble((data[columnRating]));
				Long timestamp = Long.parseLong(data[columnTimeStamp]);
				// With the new ids, we will put the the ratings to be printed in the dst file

				if (storationUsers.get(idUserTransformed) == null) { // New user
					Map<Long, List<Tuple2<Double, Long>>> map = new TreeMap<Long, List<Tuple2<Double, Long>>>();
					List<Tuple2<Double, Long>> lst = new ArrayList<Tuple2<Double, Long>>();
					lst.add(new Tuple2<Double, Long>(rating, timestamp));
					map.put(idItemTransformed, lst);
					// Add item to list
					storationUsers.put(idUserTransformed, map);
				} else { // User exits
					Map<Long, List<Tuple2<Double, Long>>> map = storationUsers.get(idUserTransformed);
					List<Tuple2<Double, Long>> lst = map.get(idItemTransformed);
					if (lst == null) { // New item for the user
						lst = new ArrayList<Tuple2<Double, Long>>();
						lst.add(new Tuple2<Double, Long>(rating, timestamp));
						map.put(idItemTransformed, lst);
					} else {
						if (removeDuplicates) {
							if (lst.get(0).v2 < timestamp) { // We obtain the newest timestamp
								lst.clear();
								lst.add(new Tuple2<Double, Long>(rating, timestamp));
							}
						} else {
							lst.add(new Tuple2<Double, Long>(rating, timestamp));
						}
					}
				}
			});
			writer = new BufferedWriter(new FileWriter(dstPath));
			// print the new ratings
			for (Long user : storationUsers.keySet()) {

				for (Long item : storationUsers.get(user).keySet()) {
					for (Tuple2<Double, Long> t : storationUsers.get(user).get(item)) {
						writer.write(user + "\t" + item + "\t" + t.v1 + "\t" + t.v2 + "\n");
					}

				}
			}

			writer.close();
		} catch (Exception e) {
			// TODO: handle exception
		}

	}
	
	/***
	 * Method to perform a simple random split
	 * 
	 * @param source          the source file
	 * @param destTrain       the destination train
	 * @param destTest        the destination test
	 * @param percentajeTrain the percentage of ratings that will go to the train
	 *                        and the test set
	 */
	public static void randomSplit(String source, String destTrain, String destTest, double percentageTrain, long seed) {
		try {
			PrintStream resultFile1 = new PrintStream(destTrain);
			PrintStream resultFile2 = new PrintStream(destTest);

			Stream<String> stream = Files.lines(Paths.get(source));
			Random n = new Random();
			n.setSeed(seed);
			stream.forEach(line -> {
				if (n.nextFloat() > percentageTrain) {
					resultFile2.println(line);
				}
				else {
					resultFile1.println(line);
				}
			});
			stream.close();
			resultFile1.close();
			resultFile2.close();

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	
	public static void DatasetReductionRatings(String srcpath, String dstPath, int numberOfRatingsUser,
			int numberOfRatingsItems, int numberOfUniqueRatingsUser, int numberOfUniqueRatingsItems) {
		boolean implicit = false;
		// Assume structure of the file is: UserdId(long) ItemID(long)
		// rating(double) timestamp(long)

		// Map of users -> userID and list of ItemID-rating-timestamp
		Map<String, List<Tuple3<String, Double, String>>> users = new TreeMap<String, List<Tuple3<String, Double, String>>>();
		// Map of items -> itemID and list of UserID-rating-timestamp
		Map<String, List<Tuple3<String, Double, String>>> items = new TreeMap<String, List<Tuple3<String, Double, String>>>();

		String characterSplit = "\t";

		// Parameters to configure
		int columnUser = 0;
		int columnItem = 1;
		int columnRating = 2;
		int columnTimeStamp = 3;
		// switch to true

		PrintStream writer = null;

		try (Stream<String> stream = Files.lines(Paths.get(srcpath))) {
			stream.forEach(line -> {
				String[] data = line.split(characterSplit);
				String idUser = data[columnUser];
				String idItem = data[columnItem];
				double rating = 0;

				if (!implicit) {
					rating = Double.parseDouble((data[columnRating]));
				}
				else {
					rating = 1.0;
				}
				String timestamp;
				
				if (data.length > 3) {
					timestamp = data[columnTimeStamp];
				} else {
					timestamp = "1";
				}
				
				
				List<Tuple3<String, Double, String>> lstu = users.get(idUser);
				if (lstu == null) { // New user
					lstu = new ArrayList<>();
					// Add item to list
					users.put(idUser, lstu);
				}
				lstu.add(new Tuple3<String, Double, String>(idItem, rating, timestamp));

				List<Tuple3<String, Double, String>> lsti = items.get(idItem);
				if (lsti == null) { // New item
					lsti = new ArrayList<>();
					// Add item to list
					items.put(idItem, lsti);
				}
				lsti.add(new Tuple3<String, Double, String>(idUser, rating, timestamp));
			});
			stream.close();

			while (true) {
				if (checkStateCore(users, items, numberOfRatingsUser, numberOfRatingsItems, numberOfUniqueRatingsUser, numberOfUniqueRatingsItems)) {
					break;
				}
				updateMaps(users, items, numberOfRatingsUser, numberOfRatingsItems, numberOfUniqueRatingsUser, numberOfUniqueRatingsItems);
			}
			writer = new PrintStream(dstPath);

			// Writing the new data to the file
			for (String user : users.keySet()) {
				List<Tuple3<String, Double, String>> lst = users.get(user);
				for (Tuple3<String, Double, String> t : lst) {
					writer.println(user + "\t" + t.v1 + "\t" + t.v2 + "\t" + t.v3);
				}
			}
			writer.close();

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	
	/***
	 * Method to remove the users and items that have not a certain number of
	 * ratings
	 * 
	 * @param users              the preferences of the users
	 * @param items              the preference of the items
	 * @param minimumRatingUsers minimum ratings that the users may have
	 * @param minimumRatingItems minimum ratings that the items may have
	 */
	private static void updateMaps(Map<String, List<Tuple3<String, Double, String>>> users,
			Map<String, List<Tuple3<String, Double, String>>> items, int minimumRatingUsers, int minimumRatingItems, 
			int numberOfUniqueRatingsUser, int numberOfUniqueRatingsItems) {
		Set<String> usersWithLess = new TreeSet<>(
				users.entrySet().stream().filter(t -> t.getValue().size() < minimumRatingUsers).map(t -> t.getKey())
						.collect(Collectors.toSet()));
		Set<String> itemsWithLess = new TreeSet<>(
				items.entrySet().stream().filter(t -> t.getValue().size() < minimumRatingItems).map(t -> t.getKey())
						.collect(Collectors.toSet()));
		
		Set<String> usersWithLessUnique = new TreeSet<>(
				users.entrySet().stream().filter(t -> users.get(t.getKey()).stream().map(t3 -> t3.v1).collect(Collectors.toSet()).size() < numberOfUniqueRatingsUser).map(t -> t.getKey())
						.collect(Collectors.toSet()));
		
		Set<String> itemsWithLessUnique = new TreeSet<>(
				items.entrySet().stream().filter(t -> items.get(t.getKey()).stream().map(t3 -> t3.v1).collect(Collectors.toSet()).size() < numberOfUniqueRatingsItems).map(t -> t.getKey())
						.collect(Collectors.toSet()));
		
		
		usersWithLess.addAll(usersWithLessUnique);
		itemsWithLess.addAll(itemsWithLessUnique);

		// Now for the items we must remove the users that have rated that item
		// AND the items that have less than that number of ratings
		Set<String> totalItems = new TreeSet<>(items.keySet());
		for (String item : totalItems) {
			List<Tuple3<String, Double, String>> lst = items.get(item);
			items.put(item, new ArrayList<>(
					lst.stream().filter(t -> !usersWithLess.contains(t.v1)).collect(Collectors.toList())));
		}

		// Now remove the items from the users
		Set<String> totalUsers = new TreeSet<>(users.keySet());
		for (String user : totalUsers) {
			List<Tuple3<String, Double, String>> lst = users.get(user);
			users.put(user, new ArrayList<>(
					lst.stream().filter(t -> !itemsWithLess.contains(t.v1)).collect(Collectors.toList())));
		}

		// Now remove all items and all users that have less than the ratings
		for (String user : usersWithLess) {
			users.remove(user);
		}

		for (String item : itemsWithLess) {
			items.remove(item);
		}

		//
		totalUsers = new TreeSet<>(users.keySet());
		for (String user : totalUsers) {
			List<Tuple3<String, Double, String>> lst = users.get(user);
			users.put(user,
					new ArrayList<>(lst.stream().filter(t -> (items.get(t.v1) != null)).collect(Collectors.toList())));
		}

		//
		totalItems = new TreeSet<>(items.keySet());
		for (String item : totalItems) {
			List<Tuple3<String, Double, String>> lst = items.get(item);
			items.put(item,
					new ArrayList<>(lst.stream().filter(t -> (users.get(t.v1) != null)).collect(Collectors.toList())));
		}

		System.out.println("Iteration: " + users.size() + " " + items.size());

	}
	
	public static void implicitToExplicitMaxRatings(String trainFile, String outputFile, String charactersSplit) {
		Map<String, Integer> userMaximums = new HashMap<String, Integer>();

		int columnUser = 0;
		int columnItem = 1;
		int columnRatingImplicits = 2;
		boolean ignoreFirstLine = true;
		Stream<String> stream = null;
		try {
			if (ignoreFirstLine) {
				stream = Files.lines(Paths.get(trainFile)).skip(1);
			}
			else {
				stream = Files.lines(Paths.get(trainFile));
			}
			stream.forEach(line -> {
				String[] data = line.split(charactersSplit);
				int numberList = Integer.parseInt(data[columnRatingImplicits]);
				if (userMaximums.get(data[columnUser]) == null) {
					userMaximums.put(data[columnUser], numberList);
				}
				if (userMaximums.get(data[columnUser]) < numberList) {
					userMaximums.put(data[columnUser], numberList);
				}
			});
			stream.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		// Now, reading again and print in the output file the round dividing by the
		// largest number of listenings
		try {
			if (ignoreFirstLine) {
				stream = Files.lines(Paths.get(trainFile)).skip(1);
			}
			else {
				stream = Files.lines(Paths.get(trainFile));
			}
			final PrintStream writer = new PrintStream(outputFile);
			stream.forEach(line -> {
				String[] data = line.split(charactersSplit);
				// The new value is the actual number of listenings * 4 / maximum number of
				// listenings. As this value is between 0 and 4, we add a 1
				long newValue = Math.round(
						(Double.parseDouble(data[columnRatingImplicits]) * 4.0) / userMaximums.get(data[columnUser]))
						+ 1;
				writer.println(data[columnUser] + "\t" + data[columnItem] + "\t" + newValue);
			});
			writer.close();
			stream.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
	
	/***
	 * Method to make a global split (oldest ratings to train, newest to test)
	 * 
	 * @param srcPath      the source path
	 * @param dstPathTrain the destination of the train subset
	 * @param dstPathTest  the destination of the test subset
	 * @param trainPercent the percentage of ratings that the train subset must have
	 */
	public static void DatasetTemporalGlobalSplit(String srcPath, String dstPathTrain, String dstPathTest,
			double trainPercent) {

		List<Preference> allRatings = new ArrayList<>();
		String characterSplit = "\t";

		// Parameters to configure
		int columnUser = 0;
		int columnItem = 1;
		int columnRating = 2;
		int columnTimeStamp = 3;
		boolean ignoreFirstLine = false; // If we want to ignore first line, switch to true

		Stream<String> stream = null;
		try {
			if (ignoreFirstLine) {
				stream = Files.lines(Paths.get(srcPath)).skip(1);
			} else {
				stream = Files.lines(Paths.get(srcPath));
			}
			stream.forEach(line -> {

				String[] data = line.split(characterSplit);
				Long idUser = Long.parseLong(data[columnUser]);
				Long idItem = Long.parseLong(data[columnItem]);
				Double rating = Double.parseDouble((data[columnRating]));
				Long timeStamp = Long.parseLong(data[columnTimeStamp]);
				Preference p = new Preference(idUser, idItem, rating, timeStamp);
				allRatings.add(p);

			});
			stream.close();

			PrintStream writer = null;
			PrintStream writer2 = null;

			writer = new PrintStream(dstPathTrain);
			writer2 = new PrintStream(dstPathTest);
			// Now temporal split
			Collections.sort(allRatings);
			int indexLimit = (int) (trainPercent * allRatings.size());

			for (int i = 0; i < allRatings.size(); i++) {
				Preference p = allRatings.get(i);
				if (i < indexLimit)
					writer.println(
							p.getIdUser() + "\t" + p.getIdItem() + "\t" + p.getRating() + "\t" + p.getTimeStamp());
				else
					writer2.println(
							p.getIdUser() + "\t" + p.getIdItem() + "\t" + p.getRating() + "\t" + p.getTimeStamp());
			}

			writer2.close();
			writer.close();

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	
	static class Preference implements Comparable<Preference> {

		Long idUser;
		Long idItem;
		Double rating;
		Long timeStamp;

		public Preference(Long idUser, Long idItem, Double rating, Long timeStamp) {
			super();
			this.idUser = idUser;
			this.idItem = idItem;
			this.rating = rating;
			this.timeStamp = timeStamp;
		}

		public Long getIdUser() {
			return idUser;
		}

		public void setIdUser(Long idUser) {
			this.idUser = idUser;
		}

		public Long getIdItem() {
			return idItem;
		}

		public void setIdItem(Long idItem) {
			this.idItem = idItem;
		}

		public Double getRating() {
			return rating;
		}

		public void setRating(Double rating) {
			this.rating = rating;
		}

		public Long getTimeStamp() {
			return timeStamp;
		}

		public void setTimeStamp(Long timeStamp) {
			this.timeStamp = timeStamp;
		}

		@Override
		public int compareTo(Preference arg0) {
			return this.getTimeStamp().compareTo(arg0.getTimeStamp());
		}

	}
	
	/***
	 * Method to check if we are satisfying the k-core property
	 * 
	 * @param users              the preference of the users
	 * @param items              the preference of the items
	 * @param minimumRatingUsers minimum ratings that the users may have
	 * @param minimumRatingItems minimum ratings that the items may have
	 * @return true if the k-core is satisfied, false if not
	 */
	private static boolean checkStateCore(Map<String, List<Tuple3<String, Double, String>>> users,
			Map<String, List<Tuple3<String, Double, String>>> items, int minimumRatingUsers, int minimumRatingItems, int numberOfUniqueRatingsUser, int numberOfUniqueRatingsItem) {
		for (String user : users.keySet()) {
			if (users.get(user).size() < minimumRatingUsers) {
				return false;
			}
			
			Set<String> uniqueRatingsUser = new HashSet<>(users.get(user).stream().map(t -> t.v1).collect(Collectors.toSet()));
			if (uniqueRatingsUser.size() < numberOfUniqueRatingsUser) {
				return false;
			}
		}

		for (String item : items.keySet()) {
			if (items.get(item).size() < minimumRatingItems) {
				return false;
			}
			
			Set<String> uniqueRatingsItem = new HashSet<>(items.get(item).stream().map(t -> t.v1).collect(Collectors.toSet()));
			if (uniqueRatingsItem.size() < numberOfUniqueRatingsItem) {
				return false;
			}
		}
		return true;
	}

}
