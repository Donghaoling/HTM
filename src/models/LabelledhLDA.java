package models;

import gnu.trove.TIntIntHashMap;
import gnu.trove.TObjectDoubleHashMap;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import utils.Dirichlet;
import utils.IDSorter;
import utils.Randoms;
import utils.Utils;
//输出到文件

public class LabelledhLDA {

	ArrayList<ArrayList<Integer>> training;
	ArrayList<String> vocab;
	
    NCRPNode rootNode;
    /**
     * a tmp node
     */
    NCRPNode node;

    int numLevels;
    int numDocuments;
    int numTypes;

    double alpha = 10.0; // smoothing on topic distributions
    double gamma = 0.5; // "imaginary" customers at the next, as yet unused table
    double eta = 0.1;   // smoothing on word distributions
    double etaSum;
    
	boolean showProgress = true;
	int displayTopicsInterval =10;
	int numWordsToDisplay = 100;
	
    /**
     * hidden variable for each word, indicate its level on the topic tree
     */
    int[][] levels; // indexed < doc, token >
    NCRPNode[] documentLeaves; // currently selected path (ie leaf node) through the NCRP tree
	int totalNodes = 0;
	static String modelname = "labelled.hlda";
    Randoms random;

	/**
	 * number of categories in the first level
	 */
	static int K = 6;
	ArrayList<String> label = new ArrayList<String>();
	int labelweights = 100;
	ArrayList<String> cates = new ArrayList<String>();
	int[] labels;
	int unlabeledDocs = 0;
	double acceptRate = 0.5;
	
    public static void main (String[] args) {
		try {
			System.out.println("loading data files... ");
			//read inputs
			ArrayList<ArrayList<Integer>> training = Utils.readDataDir(args[0]);
			ArrayList<String> vocab = Utils.readLines(args[0]+"/vocab.txt");
			ArrayList<String> label = Utils.readLines(args[0]+"/doclabel.txt");
			
			System.out.println(modelname+ " start sampling...");
			
			LabelledhLDA sampler = new LabelledhLDA();
			
			sampler.initialize(training, vocab, label, 3, new Randoms(0));
			sampler.estimate(10);
			
			/*if (training != null) {
				double empiricalLikelihood = sampler.empiricalLikelihood(1000, training);
				System.out.println("Empirical likelihood: " + empiricalLikelihood);
			}*/
			
		} catch (Exception e) {
			e.printStackTrace();
		}
    }
    
    public void initialize(ArrayList<ArrayList<Integer>> training, ArrayList<String> vocab,
    		ArrayList<String> label, int numLevels, Randoms random) {
    	
		this.training = training;
		this.label = label;
		this.vocab = vocab;
		this.numLevels = numLevels;
		this.random = random;

		//get total number of docs
		numDocuments = training.size();
		
		//get total number of words
		numTypes = vocab.size();
	
		etaSum = eta * numTypes;

		//init cates
		cates.add("arts");
		cates.add("books");
		cates.add("business");
		cates.add("nyregion");
		cates.add("sports");
		cates.add("world");

		labels = new int[numDocuments];
		for (int d = 0; d<numDocuments; d++) {
			if (this.random.nextDouble() <= acceptRate)
				labels[d] = cates.indexOf(label.get(d));
			else {
				labels[d] = -1;
				unlabeledDocs++;
			}
		}
		
		// Initialize a single path, dim number of levels
		NCRPNode[] path = new NCRPNode[numLevels];

		//new the root and put all words on it
		rootNode = new NCRPNode(numTypes);

		//initialize each categories
		for (int i=0; i<K; i++) {
			rootNode.addChild();
		}

		levels = new int[numDocuments][];
		
		//create a leaves array store the map of doc to leaf node
		documentLeaves = new NCRPNode[numDocuments];

		// Initialize and fill the topic pointer arrays for 
		//  every document. Set everything to the single path that 
		//  we added earlier.
		for (int doc=0; doc < numDocuments; doc++) { //for each document
			ArrayList<Integer> fs = training.get(doc);
            //total number of words in this doc
            int seqLen = fs.size();

            //sample a path for this doc
			path[0] = rootNode;
			rootNode.customers++;
			//if the node has the document,then the value is 1
			rootNode.docID[doc]=1;
			
			//random initialize our fist level
			path[1] = rootNode.children.get(random.nextInt(K));
			path[1].customers++;
			path[1].docID[doc]=1;
			
			//start from second level
			for (int level = 2; level < numLevels; level++) {
				path[level] = path[level-1].select();
				path[level].customers++;
				path[level].docID[doc]=1;
			}
			
			//save leaf node 
			node = path[numLevels - 1];
			documentLeaves[doc] = node;
			
			//random allocated levels for each word
			levels[doc] = new int[seqLen];
			for (int token=0; token < seqLen; token++) {
				int type = fs.get(token);
				levels[doc][token] = random.nextInt(numLevels);
				node = path[ levels[doc][token] ];
				node.totalTokens++;
				node.typeCounts[type]++;
			}
		}
	}

    int count = 0;
	public void estimate(int numIterations) throws IOException {
		for (int iteration = 1; iteration <= numIterations; iteration++) {
			
			count = 0;
			for (int doc=0; doc < numDocuments; doc++) {
				samplePath(doc, iteration);
			}
			
			
			for (int doc=0; doc < numDocuments; doc++) {
				sampleTopics(doc);
			}
			
			if (showProgress) {
				System.out.print(".");
				if (iteration % 50 == 0) {
					System.out.println(" " + iteration);
				}
			}

			if (iteration % displayTopicsInterval == 0) {
				tracenode();
				printNodes();
			}
		}
		
		System.out.println("Accuracy:"+(double)count/(double)unlabeledDocs);
    }

    public void samplePath(int doc, int iteration) {
		NCRPNode[] path = new NCRPNode[numLevels];
		NCRPNode node;
		int level, token, type;

		/////////////////////////////Before sample a path
		
		//save current path of current doc leaf to path[]
		node = documentLeaves[doc];
		for (level = numLevels - 1; level >= 0; level--) {
			path[level] = node;
			node = node.parent;
		}

		//detach current path
		documentLeaves[doc].dropPath(doc);
	
		//for current doc the typeCount of each level
		TIntIntHashMap[] typeCounts = new TIntIntHashMap[numLevels];

		int[] docLevels;

		for (level = 0; level < numLevels; level++) {
			typeCounts[level] = new TIntIntHashMap();
		}

		docLevels = levels[doc];
		ArrayList<Integer> fs = training.get(doc);
	    
		// Save the counts of every word at each level, and remove
		//  counts from the current path
		for (token = 0; token < docLevels.length; token++) {
			level = docLevels[token];
			type = fs.get(token);
	    
			if (! typeCounts[level].containsKey(type)) {
				typeCounts[level].put(type, 1);
			}
			else {
				typeCounts[level].increment(type);
			}

			path[level].typeCounts[type]--;
			assert(path[level].typeCounts[type] >= 0);
	    
			path[level].totalTokens--;	    
			assert(path[level].totalTokens >= 0);
		}

		/////////////////////////////////Prepare weights
		// Calculate the level weight for unallocated mass of a new node at a given level.
		double[] newTopicWeights = new double[numLevels];
		for (level = 1; level < numLevels; level++) {  // Skip the root...
			int[] types = typeCounts[level].keys();
			int totalTokens = 0;

			for (int t: types) {
				for (int i=0; i<typeCounts[level].get(t); i++) {
					newTopicWeights[level] += 
						Math.log((eta + i) / (etaSum + totalTokens));
					totalTokens++;
				}
			}
		}
	
		//define node weight array, (multinomial distribution)   
		TObjectDoubleHashMap<NCRPNode> nodeWeights = 
			new TObjectDoubleHashMap<NCRPNode>();
	
		// Calculate p(c_m | c_{-m})
		calculateNCRP(nodeWeights, rootNode, 0.0);

		// Add weights for p(w_m | c, w_{-m}, z)
	
		// The path may have no further customers and therefore
		//  be unavailable, but it should still exist since we haven't
		//  reset documentLeaves[doc] yet...
		
		//calculate the new weights based on document level weights, nCRP node weights 
		calculateWordLikelihood(nodeWeights, rootNode, typeCounts, newTopicWeights, 0, labels[doc]);

		NCRPNode[] nodes = nodeWeights.keys(new NCRPNode[] {});
		double[] weights = new double[nodes.length];
		double sum = 0.0;
		double max = Double.NEGATIVE_INFINITY;

		// To avoid underflow, we're using log weights and normalizing the node weights so that 
		//  the largest weight is always 1.
		for (int i=0; i<nodes.length; i++) {
			//skip the root, therefore no increasing of the first level
			if (nodes[i].level == 0)
				continue;
				
			if (nodeWeights.get(nodes[i]) > max) {
				max = nodeWeights.get(nodes[i]);
			}
		}

		for (int i=0; i<nodes.length; i++) {
			//skip the root, therefore no increasing of the first level
			if (nodes[i].level == 0) {
				weights[i] = 0;
				continue;
			}
			
			weights[i] = Math.exp(nodeWeights.get(nodes[i]) - max);

			sum += weights[i];
		}

		///////////////////////////////////Sample path when weights ready
		//sample a new leaf node among all those nodes
		int r = random.nextDiscrete(weights, sum);
		node = nodes[r];
		
		if (node.level == 0) {
			System.out.println("!!!!"+weights[r]);
		}
		// If we have picked an internal node, we need to 
		//  add a new path.
		if (! node.isLeaf()) {
			node = node.getNewLeaf();
		}
		
		//put new leaf node back
		documentLeaves[doc] = node;
		
		//update nCRP related info along the new path
		node.addPath(doc);
		
		if (node.path[1] == cates.indexOf(label.get(doc))) {
			if (labels[doc]==-1)
				count++;
		}
		
		//update topic related info along the new path
		for (level = numLevels - 1; level >= 0; level--) {
			int[] types = typeCounts[level].keys();

			for (int t: types) {
				node.typeCounts[t] += typeCounts[level].get(t);
				node.totalTokens += typeCounts[level].get(t);
			}

			node = node.parent;
		}
    }

    public void calculateNCRP(TObjectDoubleHashMap<NCRPNode> nodeWeights, 
							  NCRPNode node, double weight) {
		for (NCRPNode child: node.children) {
			calculateNCRP(nodeWeights, child,
						  weight + Math.log((double) child.customers / (node.customers + gamma)));
		}

		nodeWeights.put(node, weight + Math.log(gamma / (node.customers + gamma)));
    }

    public void calculateWordLikelihood(TObjectDoubleHashMap<NCRPNode> nodeWeights, NCRPNode node, 
										TIntIntHashMap[] typeCounts, double[] newTopicWeights, int level,
										int label) {
		// First calculate the likelihood of the words at this level, given this topic.
		double nodeWeight = 0.0;
		int[] types = typeCounts[level].keys();
		int totalTokens = 0;

		for (int type: types) {
					
			for (int i=0; i<typeCounts[level].get(type); i++) {
				nodeWeight +=
					Math.log((eta + node.typeCounts[type] + i) /
							 (etaSum + node.totalTokens + totalTokens));
				totalTokens++;
			}
		}
		
		//increasing the weight of the nodes in subtree
		if (node.path[1] == label)
			nodeWeight += labelweights;

		// Propagate that weight to the child nodes
		for (NCRPNode child: node.children) {
            calculateWordLikelihood(nodeWeights, child,	typeCounts, newTopicWeights, level + 1, label);
        }

		// Finally, if this is an internal node, add the weight of a new path (internal node only)
		level++;
		while (level < numLevels) {
			nodeWeight += newTopicWeights[level];
			level++;
		}

		nodeWeights.adjustValue(node, nodeWeight);
    }

    /** Propagate a topic weight to a node and all its children.
		weight is assumed to be a log.
	*/
    public void propagateTopicWeight(TObjectDoubleHashMap<NCRPNode> nodeWeights,
									 NCRPNode node, double weight) {
		if (! nodeWeights.containsKey(node)) {
			// calculating the NCRP prior proceeds from the
			//  root down (ie following child links),
			//  but adding the word-topic weights comes from
			//  the bottom up, following parent links and then 
			//  child links. It's possible that the leaf node may have
			//  been removed just prior to this round, so the current
			//  node may not have an NCRP weight. If so, it's not 
			//  going to be sampled anyway, so ditch it.
			return;
		}
	
		for (NCRPNode child: node.children) {
			propagateTopicWeight(nodeWeights, child, weight);
		}

		nodeWeights.adjustValue(node, weight);
    }

    /**
     * sampling topic assignments for selected path
     * to avoid duplicated topics, considering extent this step to HDP
     * first construct some global topics
     * then selecting a global topic for each level
     * 
     * @param doc
     */
    public void sampleTopics(int doc) {
    	
    	ArrayList<Integer> fs = training.get(doc);
		int seqLen = fs.size();
		int[] docLevels = levels[doc];
		
		int[] levelCounts = new int[numLevels];
		int type, token, level;
		double sum;

		// Get the leaf
		NCRPNode node;
		node = documentLeaves[doc];

		//get path
		NCRPNode[] path = new NCRPNode[numLevels];
		for (level = numLevels - 1; level >= 0; level--) {
			path[level] = node;
			node = node.parent;
		}

		double[] levelWeights = new double[numLevels];

		// Initialize level counts
		for (token = 0; token < seqLen; token++) {
			levelCounts[ docLevels[token] ]++;
		}

		//very similar to LDA gibbs
		for (token = 0; token < seqLen; token++) { //for each word
			type = fs.get(token);
	    
			//remove selected word from its topic
			levelCounts[ docLevels[token] ]--;
			node = path[ docLevels[token] ];
			node.typeCounts[type]--;
			node.totalTokens--;
	    
			//calc weight for each topic (nodes on the path)
			//to avoid sparsity, alpha should be large
			sum = 0.0;
			for (level=0; level < numLevels; level++) {
				levelWeights[level] = 
					(alpha + levelCounts[level]) * 
					(eta + path[level].typeCounts[type]) /
					(etaSum + path[level].totalTokens);
				sum += levelWeights[level];
			}
			//sample a topic
			level = random.nextDiscrete(levelWeights, sum);

			//put word back
			docLevels[token] = level;
			levelCounts[ docLevels[token] ]++;
			node = path[ level ];
			node.typeCounts[type]++;
			node.totalTokens++;
		}
    }

	/**
	 *  Writes the current sampling state to the file specified in <code>stateFile</code>.
	 */
	public void printState() throws IOException, FileNotFoundException {
		printState(new PrintWriter(new BufferedWriter(new FileWriter(modelname))));
	}

	/**
	 *  Write a text file describing the current sampling state. 
	 */
    public void printState(PrintWriter out) throws IOException {
		int doc = 0;

		for (ArrayList<Integer> fs: training) {
			int seqLen = fs.size();
			int[] docLevels = levels[doc];
			NCRPNode node;
			int type, token, level;

			StringBuffer path = new StringBuffer();
			
			// Start with the leaf, and build a string describing the path for this doc
			node = documentLeaves[doc];
			for (level = numLevels - 1; level >= 0; level--) {
				path.append(node.nodeID + " ");
				node = node.parent;
			}

			for (token = 0; token < seqLen; token++) {
				type = fs.get(token);
				level = docLevels[token];
				
				// The "" just tells java we're not trying to add a string and an int
				out.println(path + "" + type + " " + vocab.get(type) + " " + level + " ");
			}
			doc++;
		}
	}	    


    public void printNodes() throws IOException {
		printNode(rootNode, 0);
    }

    public void printNode(NCRPNode node, int indent) throws IOException {
		StringBuffer out = new StringBuffer();
//		StringBuffer outfile = new StringBuffer();
		File file = new File("E:\\HTM_results\\results2.txt");
//		File wordfile = new File("E:\\HTM_results\\words2.txt");
		if(!file.exists())
			file.createNewFile();
//		if(!wordfile.exists())
//			wordfile.createNewFile();
		FileOutputStream fout = new FileOutputStream(file,true);
//		FileOutputStream foutwords = new FileOutputStream(file,true);
		for (int i=0; i<indent; i++) {
			out.append("  ");
//			outfile.append("  ");
		}
		out.append(node.totalTokens + "/" + node.customers + " ");//单词数和文章数
//		outfile.append(node.totalTokens + "/" + node.customers + " ");
		if(indent==1){//output the categary of the node
			String sdoc = node.getTopDocs(1);
			int docnum=Integer.parseInt(node.getTopDocs(1).replace(" ",""));
			out.append(cates.get(cates.indexOf(label.get(docnum))));
			out.append(" ");			
		}
		out.append(node.getTopWords(numWordsToDisplay));
//		outfile.append(node.getTopWords(numWordsToDisplay));
		//out.append(node.getTopWords(node.totalTokens));
		if(node.children.size()==0){
			out.append(node.getTopDocs(4));
			node.getTopicWords();
		}
		System.out.println(out);
		out.append("\n");
//		outfile.append("\n");
		fout.write(out.toString().getBytes());
//		foutwords.write(outfile.toString().getBytes());
		fout.close();
//		foutwords.close();
		
	
		for (NCRPNode child: node.children) {
			printNode(child, indent + 1);
		}
    }

    /** For use with empirical likelihood evaluation: 
     *   sample a path through the tree, then sample a multinomial over
     *   topics in that path, then return a weighted sum of words.
     */
    public double empiricalLikelihood(int numSamples, ArrayList<ArrayList<Integer>> testing)  {
		NCRPNode[] path = new NCRPNode[numLevels];
		NCRPNode node;
		
		path[0] = rootNode;

		ArrayList<Integer> fs;
		int sample, level, type, token, doc, seqLen;

		Dirichlet dirichlet = new Dirichlet(numLevels, alpha);
		double[] levelWeights;
		//dictionary
		double[] multinomial = new double[numTypes];

		double[][] likelihoods = new double[ testing.size() ][ numSamples ];
		
		//for each sample
		for (sample = 0; sample < numSamples; sample++) {
			Arrays.fill(multinomial, 0.0);

			//select a path
			for (level = 1; level < numLevels; level++) {
				path[level] = path[level-1].selectExisting();
			}
	    
			//sample level weights
			levelWeights = dirichlet.nextDistribution();
	    
			//for each words in dictionary
			for (type = 0; type < numTypes; type++) {
				//for each topic
				for (level = 0; level < numLevels; level++) {
					node = path[level];
					multinomial[type] +=
						levelWeights[level] * 
						(eta + node.typeCounts[type]) /
						(etaSum + node.totalTokens);
				}

			}

			//convert to log
			for (type = 0; type < numTypes; type++) {
				multinomial[type] = Math.log(multinomial[type]);
			}

			//calculate document likelihoods  
			for (doc=0; doc<testing.size(); doc++) {
                fs = testing.get(doc);
                seqLen = fs.size();
                
                for (token = 0; token < seqLen; token++) {
                    type = fs.get(token);
                    likelihoods[doc][sample] += multinomial[type];
                }
            }
		}
	
        double averageLogLikelihood = 0.0;
        double logNumSamples = Math.log(numSamples);
        for (doc=0; doc<testing.size(); doc++) {
        	
        	//find the max for normalization, avoid overflow of sum
            double max = Double.NEGATIVE_INFINITY;
            for (sample = 0; sample < numSamples; sample++) {
                if (likelihoods[doc][sample] > max) {
                    max = likelihoods[doc][sample];
                }
            }

            double sum = 0.0;
            //normalize 
            for (sample = 0; sample < numSamples; sample++) {
                sum += Math.exp(likelihoods[doc][sample] - max);
            }

            //calc average
            averageLogLikelihood += Math.log(sum) + max - logNumSamples;
        }

		return averageLogLikelihood;
    }
    
    public void tracenode(){  	
    	for(int nnode=0;nnode<documentLeaves.length;nnode++){
    		for(int doc=0;doc<numDocuments;doc++){
    			if(documentLeaves[nnode].docID[doc]==1){
    				docPossibilityofLeafTopic(doc,documentLeaves[doc]);  
    			}
    		}
    					   	
    	}
    	/*if(node.children.size()==0){
    		for(int doc=0;doc<numDocuments;doc++){
    			if(node.docID[doc]==1){
    				docPossibilityofLeafTopic(doc,node);
    			}
    		}
    	}
    	else{
    		for(int i=0;i<node.customers;i++){
    			tracenode(node.children.get(i));
    		}
    	}*/
    }
    public void docPossibilityofLeafTopic(int doc,NCRPNode leafnode){
    	ArrayList<Integer> fs = training.get(doc);
		int seqLen = fs.size();
		
		
		
		int type, token, level;
		double sum;

		NCRPNode node=leafnode;

		//get path
		NCRPNode[] path = new NCRPNode[numLevels];
		for (level = numLevels - 1; level >= 0; level--) {
			path[level] = node;
			node = node.parent;
		}

		double[] levelWeights = new double[numLevels];
		for(int i=0;i<numLevels;i++){
			levelWeights[i]=1.0;
		}
	
		sum=0.0;
		for (level = 0;level < numLevels; level++ ) { //for each word	    
			//calc weight for each topic (nodes on the path)
			//to avoid sparsity, alpha should be large
			
			for (token=0;token < seqLen; token++) {
				type = fs.get(token);
				levelWeights[level] = 
						levelWeights[level]+Math.log(
					(eta + path[level].typeCounts[type]) /
					(etaSum + path[level].totalTokens));
				//sum += levelWeights[level];
			}
			sum=sum+levelWeights[level];
			
		}
		leafnode.docID[doc]=Math.abs(levelWeights[numLevels-1])/Math.abs(sum);
		
		
		
    }
   
    class NCRPNode {
    	/**
    	 * number of docs on this node(topic), including docs from its children.
    	 */
		int customers;
		
		/**
		 * number of sub topics
		 */
		ArrayList<NCRPNode> children;
		NCRPNode parent;
		/**
		 * level of this node
		 */
		int level;

		/**
		 * number of words attached on this node(topic), words of children excluded.
		 */
		int totalTokens;
		
		/**
		 * for each word, counts of its appearance of current node, words of children excluded 
		 */
		int[] typeCounts;
		int[] path;

		/**
		 * ID
		 */
		public int nodeID;
		
		//index is doc ID,value is docPossibility
		public double[] docID;


		public NCRPNode(NCRPNode parent, int dimensions, int level) {
			customers = 0;
			this.parent = parent;
			children = new ArrayList<NCRPNode>();
			
			if (parent != null) {
				path = parent.path.clone();
				path[level] = parent.children.size();
			} else {
				path = new int[numLevels];
				Arrays.fill(path, -1);
				path[0] = 0;
			}
			
			this.level = level;
			totalTokens = 0;
			typeCounts = new int[dimensions];

			nodeID = totalNodes;
			totalNodes++;
			
			docID=new double[numDocuments];
			for(int i=0;i<numDocuments;i++){
				docID[i]=0;
			}
		}

		public NCRPNode(int dimensions) {
			this(null, dimensions, 0);
		}

		public NCRPNode addChild() {
			NCRPNode node = new NCRPNode(this, typeCounts.length, level + 1);
			children.add(node);
			return node;
		}

		public boolean isLeaf() {
			return level == numLevels - 1;
		}

		public NCRPNode getNewLeaf() {
			NCRPNode node = this;
			for (int l=level; l<numLevels - 1; l++) {
				node = node.addChild();
			}
			return node;
		}

		public void dropPath(int doc) {
			NCRPNode node = this;
			node.customers--;
			node.docID[doc]=0;
			if (node.customers == 0) {
				node.parent.remove(node);
			}
			for (int l = 1; l < numLevels; l++) {
				node = node.parent;
				node.customers--;
				node.docID[doc]=0;
				if (node.customers == 0) {
					node.parent.remove(node);
				}
			}
		}

		public void remove(NCRPNode node) {
			children.remove(node);
		}

		public void addPath(int doc) {
			NCRPNode node = this;
			node.customers++;
			node.docID[doc]=1;
			for (int l = 1; l < numLevels; l++) {
				node = node.parent;
				node.customers++;
				node.docID[doc]=1;
			}
		}

		public NCRPNode selectExisting() {
			double[] weights = new double[children.size()];
	    
			int i = 0;
			for (NCRPNode child: children) {
				weights[i] = (double) child.customers  / (gamma + customers);
				i++;
			}

			int choice = random.nextDiscrete(weights);
			return children.get(choice);
		}
		
		/**
		 * select a child with higher weight
		 * @return a selected child
		 */
		public NCRPNode select() {
			
			//dim number of children + 1 (unallocated mass) 
			double[] weights = new double[children.size() + 1];
	    
			//weight of unallocated probability mass
			weights[0] = gamma / (gamma + customers);
			
			//calc weight for each child based on the number of customers on them
			int i = 1;
			for (NCRPNode child: children) {
				weights[i] = (double) child.customers / (gamma + customers);
				i++;
			}

			//sample a child with higher weight
			int choice = random.nextDiscrete(weights);
			//if unallocated mass is sampled, create a new child
			if (choice == 0) {
				return(addChild());
			}
			else {
				return children.get(choice - 1);
			}
		}
	
		public String getTopWords(int numWords) {
			IDSorter[] sortedTypes = new IDSorter[numTypes];
	    
			for (int type=0; type < numTypes; type++) {
				sortedTypes[type] = new IDSorter(type, typeCounts[type]);
			}
			Arrays.sort(sortedTypes);

			StringBuffer out = new StringBuffer();
			for (int i=0; i<numWords; i++) {
				out.append(vocab.get(sortedTypes[i].getID()) + " ");
				out.append(sortedTypes[i].getWeight()+" ");
			}
			String s=out.toString();
			/*try {  
	            //文件生成路径  
	            PrintStream ps=new PrintStream("D:\\role.txt");  
	            System.setOut()  
	        } catch (FileNotFoundException e) {  
	            e.printStackTrace();  
	        } */
			/*try {
				FileOutputStream out1 = new FileOutputStream("test.txt",true);
			} catch (FileNotFoundException e) {
				// 
				e.printStackTrace();
			}*/
			return s;
		}	
		
		public void getTopicWords() throws IOException{
			/*String s="";
			StringBuffer out = new StringBuffer();
			for (int i=0; i<vocab.size(); i++) {
				//out.append(typeCounts[i] + " ");
				s=s+String.valueOf(typeCounts[i])+" ";
			}
			
			//s=out.toString();
			System.out.print(s);
			return s;*/
			FileWriter out1 = new FileWriter("E:\\HTM__results\\words5.txt",true);
			for(int i=0;i<vocab.size();i++){
				out1.write(typeCounts[i]+" ");
				//System.out.print(node.typeCounts[i]+" ");
			}
			out1.write("\n");
			out1.close();
		}

		public String getTopDocs(int numDocstoDisplay){
			NCRPNode node = this;
			int numDocs=node.customers;
			Map<Integer,Double> doc_possibility=new HashMap<Integer,Double>(numDocs);
			for (int doc=0; doc <numDocuments; doc++) {
				if(node.docID[doc]!=0){
					doc_possibility.put(doc, node.docID[doc]);
				}
			}
			List<Map.Entry<Integer,Double>> list=new ArrayList<>();  
			list.addAll(doc_possibility.entrySet());  
			Collections.sort(list, new Comparator<Map.Entry<Integer,Double>>() {   
	            public int compare(Map.Entry<Integer,Double> o1, Map.Entry<Integer,Double> o2) {   
	                return (int)(o1.getValue()-o2.getValue());   
	            }   
			});
			StringBuffer out = new StringBuffer();
			int minDocs=Math.min(numDocstoDisplay,numDocs);
			for(int i=0;i<minDocs;i++){
				out.append(list.get(i).getKey()+" ");
			}
			
			
			return out.toString();
			
		}
		
    }
}
