package utils;

import java.io.*;
import java.util.*;

/**
 * Static methods for reading/writing files
 * 
 * @author andrzeje
 *
 */
public class Utils 
{	
    public static String intArrayToString(int[] a)
    {
        StringBuilder retval = new StringBuilder();
        for(int val : a)
            {
                retval.append(String.format("%d ", val));
            }
		
        return retval.toString();
    }
	
    /**
     * Dump Collection to file, one item per line
     * @param values
     * @param outname
     * @throws IOException
     */
    public static void writeLines(Collection<String> values, String outname)
        throws IOException
    {
        FileWriter out = new FileWriter(new File(outname));
        for(String val : values)
            {				
                out.write(String.format("%s\n", val));
            }
        out.close();
    }
			
    /**
     * Write these integers out to plaintext file
     * 
     * @param filename
     * @param vals
     * @throws IOException
     */
    public static void writeIntFile(String filename, List<Integer> vals) 
        throws IOException
    {
        FileWriter out = new FileWriter(new File(filename));
        int i = 0;
        for(Integer val : vals)
            {
                out.write(String.format("%d ", val));
                i += 1;
            }
        out.close();
    }
	
    /**
     * Just return each line of the file as String
     * 
     * @param filename
     * @return
     */
	public static ArrayList<String> readLines(String filename) {
		ArrayList<String> retval = new ArrayList<String>();
		try {
			
			BufferedReader in = new BufferedReader(new InputStreamReader(
					new FileInputStream(filename), "cp936"));
			String curLine = in.readLine();
			while (curLine != null) {
				retval.add(curLine.trim());
				curLine = in.readLine();
			}
			return retval;
		} catch (IOException ioe) {
			System.out.println(String.format("Bad file(name): %s\n",
					ioe.toString()));
			return null;
		}
	}
	
	
	public static ArrayList<ArrayList<Integer>> readDataDir(String dirname) {
		
		ArrayList<ArrayList<Integer>> vals = new ArrayList<ArrayList<Integer>>(); //documents and words
		
		Scanner words;
		Scanner docs;
		
		try {
			//load type and index arrays
			words = new Scanner(new File(dirname+"/words.txt"));
			docs = new Scanner(new File(dirname+"/docs.txt"));
			vals.add(new ArrayList<Integer>());
			
			while (docs.hasNextInt()) {
							
				int idx = docs.nextInt();
				
				//expend array when we found a bigger index
				if (idx+1 > vals.size())
					for (int i = vals.size(); i<idx+1; i++)
						vals.add(new ArrayList<Integer>());
				
				//add a new
				vals.get(idx).add(words.nextInt());
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
		
		words.close();
		docs.close();
		
		return vals;
	}
	
    /**
     * Read a plaintext file containing a sequence of integers
     * 
     * @param filename 
     * @return
     */
	public static Vector<Integer> readIntFile(String filename) {

		Vector<Integer> vals = new Vector<Integer>();

		Scanner scanner;
		try {
			scanner = new Scanner(new File(filename));

			while (scanner.hasNextInt()) {
				vals.add(scanner.nextInt());
			}

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println(String.format("Bad file(name): %s\n",
					e.toString()));
			return null;
		}
		scanner.close();

		return vals;

	}

    /**
     * Parse line of doubles
     * 
     * @param line
     * @return nothing - but adds elements to vec
     */
    public static Vector<Double> parseDoubleLine(String line)
    {
        Vector<Double> vec = new Vector<Double>();
        StringTokenizer stok = new StringTokenizer(line);
        while(stok.hasMoreTokens())
            vec.add(Double.parseDouble(stok.nextToken()));
        return vec;
    }

    /**
     * Read entire file as a single String
     * 
     * @param filename
     * @return
     * @throws IOException
     */
    public static String fileSlurp(String filename) throws IOException
    {
        StringBuilder fileContent = new StringBuilder();
        BufferedReader in = new BufferedReader(new FileReader(filename));
        String line = in.readLine();
        while(line != null)
            {				
                fileContent.append(line);
                fileContent.append(" ");
                line = in.readLine();
            }			
        in.close();
        return fileContent.toString();
    }

    /**
     * Write a single String out to file
     * 
     * @param filename
     * @param output
     * @throws IOException
     */
    public static void fileSpit(String filename, String output) throws IOException
    {
        FileWriter out = new FileWriter(filename);
        out.write(output);
        out.close();
    }
    
    
}
