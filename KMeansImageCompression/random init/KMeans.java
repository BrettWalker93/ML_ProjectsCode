/*** Authors :Vibhav Gogate, 
 * 			  Brett Walker (functions kmeans() and eu_dist())
The University of Texas at Dallas

*****/


import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.lang.Math; 
import java.util.Random;
import java.util.Vector;
import java.util.Arrays;

public class KMeans {
    public static void main(String [] args){
	if (args.length < 3){
	    System.out.println("Usage: Kmeans <input-image> <k> <output-image>");
	    return;
	}
	try{
	    BufferedImage originalImage = ImageIO.read(new File(args[0]));
	    int k=Integer.parseInt(args[1]);
	    BufferedImage kmeansJpg = kmeans_helper(originalImage,k);
	    ImageIO.write(kmeansJpg, "jpg", new File(args[2])); 
	    
	}catch(IOException e){
	    System.out.println(e.getMessage());
	}	
    }
    
    private static BufferedImage kmeans_helper(BufferedImage originalImage, int k){
	int w=originalImage.getWidth();
	int h=originalImage.getHeight();
	BufferedImage kmeansImage = new BufferedImage(w,h,originalImage.getType());
	Graphics2D g = kmeansImage.createGraphics();
	g.drawImage(originalImage, 0, 0, w,h , null);
	// Read rgb values from the image
	int[] rgb=new int[w*h];
	int count=0;
	for(int i=0;i<w;i++){
	    for(int j=0;j<h;j++){
		rgb[count++]=kmeansImage.getRGB(i,j);
	    }
	}
	// Call kmeans algorithm: update the rgb values
	kmeans(rgb,k);

	// Write the new rgb values to the image
	count=0;
	for(int i=0;i<w;i++){
	    for(int j=0;j<h;j++){
		kmeansImage.setRGB(i,j,rgb[count++]);
	    }
	}
	return kmeansImage;
    }

    // Update the array rgb by assigning each entry in the rgb array to its cluster center
    private static void kmeans(int[] rgb, int k){
    	
    	//randomly assign centers
    	int[] centers = new int[k];

    	Vector<Vector<Integer>> clusters = new Vector<Vector<Integer>>(k);
    	
    	int[] sorted = rgb.clone();
    	Arrays.sort(sorted);
    	
    	long avg = 0;
    	    	
    	Random rand = new Random();
    	
    	for (int i = 0; i < k; i++) {
    		//random init
    		centers[i] = rand.nextInt();
    		
    		//slice init
    		//centers[i] = sorted[(i + 1) * rgb.length / (k+1)];
    		
    	}
    	
    	//for (int i = 0; i < rgb.length; i++) {
    	//	System.out.println(Integer.toString(rgb[i]));
    	//}
    	
    	
    	//show initial centers
    	//System.out.println("Initial centers: " + Arrays.toString(centers));
    	
    	boolean flag = true;
    	int iter = 0;
    	while(flag) {
    		
    		for (int i = 0; i < k; i++) {
    			clusters.add(new Vector<Integer>());
    		}
    		
    		//assign clusters
    		for (int i = 0; i < rgb.length; i++) {
    			double min_dist = Double.MAX_VALUE;
    			int min_center_index = 0;
    			for (int j = 0; j < k; j++) {
    				
    				//distance
    				double dist = sq_eu_dist(rgb[i], centers[j]);

    				//set center
    				if (dist < min_dist) {
    					min_dist = dist;
    					min_center_index = j;
    				}
    				
    			}
    			
    			clusters.get(min_center_index).add(i);
    			
    		}
    		
    		int[] new_centers = new int[k];
    		
    		for (int j = 0; j < k; j++) {
    			
    			new_centers[j] = 0;
    			
    			double new_r = 0;
    			double new_g = 0;
    			double new_b = 0;
    			
    			for (int e : clusters.get(j)) {
    				int color = rgb[e];
    				new_r += (color >> 16) & 0xFF;
    				new_g += (color >> 8) & 0xFF;
    				new_b += color & 0xFF;
    			}
    			
    			int size = clusters.get(j).size();
    			if (size == 0) {
    				size = 1;
    			}
    			
    			new_r /= size;
    			new_g /= size;
    			new_b /= size;
    			
    			int int_new_r = (int)new_r;
    			int int_new_g = (int)new_g;
    			int int_new_b = (int)new_b;
    			
    			new_centers[j] = (0xFF << 24) | (int_new_r << 16) | (int_new_g << 8) | int_new_b;
    		}
    		
    		double iter_dist = 0.0; 
    		
    		System.out.println("After iterations: " + Integer.toString(iter));
    		System.out.println("Centers: " + Arrays.toString(centers));
    		System.out.println("With sizes:");
    		    		
    		for (int j = 0; j < k; j++) {
    			
    			iter_dist += sq_eu_dist(new_centers[j], centers[j]);
    			centers[j] = new_centers[j];
    			
    			System.out.println(Integer.toString(clusters.get(j).size()));
    		}
    		
    		
    		iter_dist /= k;
    		
    		iter++;
    		if (iter > 99) {
    			flag = false;
    		}
    		if (iter_dist < 0.01) {
    			flag = false;
    		}
    	}   	  	
    	

		for (int j = 0; j < k; j++) { 
			for (int ind : clusters.get(j))
				rgb[ind] = centers[j];
		}
    		
    		
    	       	
    }
    
    private static double sq_eu_dist(int m, int n) {
    	double dist = 0.0;

		double r = (double)((m >> 16) & 0xFF);
		double g = (double)((m >> 8) & 0xFF);
		double b = (double)(m & 0xFF);

		double r_c = (double)((n >> 16) & 0xFF);
		double g_c = (double)((n >> 8) & 0xFF);
		double b_c = (double)(n & 0xFF);
		
		dist = (r - r_c) * (r - r_c) + (g - g_c) * (g - g_c) + (b - b_c) * (b - b_c);
		
    	return dist;
    	
    }

}
