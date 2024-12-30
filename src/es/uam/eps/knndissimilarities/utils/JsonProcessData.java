package es.uam.eps.knndissimilarities.utils;

import java.io.FileReader;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

/**
 * Class to process JSON files
 * @author Pablo Sanchez
 *
 */
public final class JsonProcessData {
	
	/***
	 * process JSON for the goodreads dataset (https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/reviews)
	 * goodreads_reviews_spoiler.json.gz
	 * @param jsonFile
	 * @param outputFile
	 */
	public static void processJSONGoodReads(String jsonFile, String outputFile) {
		JSONParser parser = new JSONParser();
		
		SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
		
		try {
			PrintStream out = new PrintStream(outputFile);

			Stream<String> stream = Files.lines(Paths.get(jsonFile));
			stream.forEach(line -> {
				
				Object obj;
				try {
					obj = parser.parse(line);
					JSONObject interaction = (JSONObject) obj;
					
					    
					String userID = (String) interaction.get("user_id");
					String bookID = (String) interaction.get("book_id");
					Long rating = (Long) interaction.get("rating");
					String date = (String) interaction.get("timestamp");
					Long dateTime = dateFormat.parse(date).getTime() / 1000;
					    
					out.println(userID + "\t" + bookID + "\t" + rating+ "\t" + dateTime);  
				} catch (ParseException | java.text.ParseException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
		        
				
			});
			out.close();
			stream.close();
			
			
			
		} catch (Exception e) {
			e.printStackTrace();
		} 
	}
	
	public static void processJSONEpisions(String jsonFile, String outputFile) {
		JSONParser parser = new JSONParser();
				
		try {
			PrintStream out = new PrintStream(outputFile);

			Stream<String> stream = Files.lines(Paths.get(jsonFile));
			AtomicInteger c = new AtomicInteger(1);
			stream.forEach(line -> {
				
				Object obj;
				try {
					obj = parser.parse(line);
					JSONObject interaction = (JSONObject) obj;
					
					    
					String userID = (String) interaction.get("user");
					String productID = (String) interaction.get("item");
					Double rating = (Double) interaction.get("stars");
					Long timestamp = (Long) interaction.get("time");
					    
					out.println(userID + "\t" + productID + "\t" + rating+ "\t" + timestamp);  
					c.addAndGet(1);

				} catch (ParseException e) {
					// TODO Auto-generated catch block
					System.out.println(line);
					System.out.println(c);
					e.printStackTrace();
				}
		        
				
			});
			out.close();
			stream.close();
			
			
			
		} catch (Exception e) {
			e.printStackTrace();
		} 
	}

}
