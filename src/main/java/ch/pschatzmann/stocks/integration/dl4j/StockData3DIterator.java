package ch.pschatzmann.stocks.integration.dl4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.ta4j.core.Indicator;
import org.ta4j.core.num.Num;

import ch.pschatzmann.stocks.accounting.IHistoricValue;
import ch.pschatzmann.stocks.integration.HistoricValues;

/**
 * IStockRecordIterator using a 3d ND4j Array which is suited to be used by an
 * RNN network. 
 */

public class StockData3DIterator implements DataSetIterator {
	private static final long serialVersionUID = 1L;
	private int currentIndex = 0;
	private int inputSize; // number of features for a stock data
	private int miniBatchSize; // mini-batch size
	private int outcomeSize; // default 1, say, one day ahead prediction
	private DataSetPreProcessor preProcessor;
	private IteratorData data;
	private int inLength;
	private int outLength;
	private int slidingWindow = 1;
	private boolean masking = true; // support for masked data 
	private List<Integer> indexes; // index positions used for shuffling 
	private boolean scalingPerDataset = false;
	private List<Long> inScalerPositionsForOut;
	private boolean fitLabel = true;
	private boolean hasNext = true;

	/**
	 * Trains one week of perdiction based on 3 months of history
	 * @param in
	 * @param out
	 */
	public StockData3DIterator(List<Indicator<Num>> in, List<Indicator<Num>> out ) {
		this(in,out,1,60,4,1);
	}

	/**
	 * Constructor with all parameters
	 * 
	 * @param id
	 * @param miniBatchSize
	 * @param exampleLength
	 * @param splitRatio
	 * @param category
	 */
	public StockData3DIterator(List<Indicator<Num>> inInd, List<Indicator<Num>> outInd, int batchSize,
			int inLength, int outLength, int slidingWindow) {
		this.data = new IteratorData(inInd, outInd);
		inScalerPositionsForOut = LongStream.range(0, outInd.size()).mapToObj(l -> l).collect(Collectors.toList());
		this.miniBatchSize = Math.max(batchSize, 1);
		this.outcomeSize = data.getOut().size();
		this.inputSize = data.getIn().size();
		this.inLength = inLength;
		this.outLength = outLength;
		this.slidingWindow = Math.max(1,slidingWindow);
		List<Integer> intList = IntStream.range(0, data.getSize()).boxed().collect(Collectors.toList());
		this.indexes = new ArrayList(intList);
		reset();
	}

	@Override
	public DataSet next(int num) {
		this.hasNext = false;
		INDArray input;
		INDArray label;
		INDArray inMask = null;
		INDArray outMask = null;
		
		if (this.isMasking()) {
			int maxLength = Math.max(inLength, outLength);
			input = Nd4j.zeros(new int[] { miniBatchSize, inputSize, maxLength }, 'f');
			label = Nd4j.zeros(new int[] { miniBatchSize, outcomeSize, maxLength }, 'f');
			inMask = Nd4j.zeros(new int[] { miniBatchSize, maxLength }, 'f');
			outMask = Nd4j.zeros(new int[] { miniBatchSize, maxLength }, 'f');
		} else {
			input = Nd4j.zeros(new int[] { miniBatchSize, inputSize, inLength }, 'f');
			label = Nd4j.zeros(new int[] { miniBatchSize, outcomeSize, outLength }, 'f');			
		}

		for (int batchIndex = 0; batchIndex < num; batchIndex++) {		
			int startPos = currentIndex;
			// setup input
			for (int dayIndex = 0; dayIndex < this.inLength; dayIndex++) {
				int featurePos = 0;
				for (HistoricValues historicValueFeature : data.getIn()) {
					List<IHistoricValue> list = historicValueFeature.list();
					if (dayIndex+startPos < list.size()) {
						this.hasNext = true;
						input.putScalar(new int[] { batchIndex, featurePos, dayIndex },(list.get(getPos(dayIndex+startPos)).getValue()));
						if (inMask!=null)
							inMask.putScalar(new int[] { batchIndex, dayIndex }, 1.0);
					}
					featurePos++;
				}
			}	
			// setup output
			for (int dayIndex = 0; dayIndex < this.outLength; dayIndex++) {
				int labelPos = 0;
				for (HistoricValues historicValueLabel : data.getOut()) {
					List<IHistoricValue> list = historicValueLabel.list();
					int dayIndexAbs = dayIndex-this.outLength+this.inLength;
					if (dayIndexAbs+startPos < list.size()) {
						this.hasNext = true;
						label.putScalar(new int[] { batchIndex, labelPos, dayIndexAbs},(list.get(getPos(dayIndexAbs+startPos)).getValue()));
						if (outMask!=null)
							outMask.putScalar(new int[] { batchIndex, dayIndexAbs }, 1.0);
					}
					labelPos++;
				}
			}
			currentIndex+=this.slidingWindow;
		
		}
		DataSet result = null;
		if (this.isScalingPerDataset()) {
			NormalizerMinMaxScalerLabelsFromFeatures norm = new NormalizerMinMaxScalerLabelsFromFeatures(inScalerPositionsForOut);
			norm.fitLabel(this.fitLabel);
			result = isMasking() ?  new ScalingDataSet(input, label, inMask, outMask, norm ):new DataSet(input, label);
		} else {
			result = isMasking() ?  new DataSet(input, label, inMask, outMask ):new DataSet(input, label);
			if (this.getPreProcessor() != null) {
				this.getPreProcessor().preProcess(result);
			}
		}

		return result;
	}

	@Override
	public boolean hasNext() {
		return hasNext;
	}
	
	public int currentIndex() {
		return currentIndex;
	}
	
	public int maxIndex() {
		return this.data.getSize() - inLength;
	}

	@Override
	public DataSet next() {
		return next(this.miniBatchSize);
	}

	@Override
	public int inputColumns() {
		return inputSize;
	}

	@Override
	public int totalOutcomes() {
		return this.outcomeSize;
	}
	
	public int inputPeriods() {
		return inLength;
	}
	
	public int outcomePeriods() {
		return outLength;
	}

	@Override
	public boolean resetSupported() {
		return true;
	}

	@Override
	public boolean asyncSupported() {
		return false;
	}

	@Override
	public void reset() {
		currentIndex = 0;
		hasNext = true;
	}

	@Override
	public int batch() {
		return this.miniBatchSize;
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		this.preProcessor = preProcessor;
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		return preProcessor;
	}

	@Override
	public List<String> getLabels() {
		return data.getLabels();
	}

	public boolean isMasking() {
		return masking;
	}

	public void setMasking(boolean masking) {
		this.masking = masking;
	}
	
	/**
	 *  Shuffle the data: we just shuffle the indexes that are used to access the data
	 */
	public void shuffle() {
		Collections.shuffle(indexes);
	}
	
	/**
	 * Returns the shuffeled position 
	 * @param pos
	 * @return
	 */
	protected int getPos(int pos) {
		return indexes.get(pos);
	}

	/**
	 * Scaling per dataset
	 * @return
	 */
	public boolean isScalingPerDataset() {
		return scalingPerDataset;
	}

	/**
	 * Defines that the scaling should run for each dataset separatly
	 * @param scalingPerDataset
	 */
	public void setScalingPerDataset(boolean scalingPerDataset) {
		this.scalingPerDataset = scalingPerDataset;
	}

	public boolean isFitLabel() {
		return fitLabel;
	}

	public void setFitLabel(boolean fitLabel) {
		this.fitLabel = fitLabel;
	}
	
	public List<Date> getDates() {
		return this.data.getDates();
	}

	public List<Long> getInScalerPositionsForOut() {
		return inScalerPositionsForOut;
	}

	public void setInScalerPositionsForOut(List<Long> inScalerPositionsForOut) {
		this.inScalerPositionsForOut = inScalerPositionsForOut;
	}


}
