package ch.pschatzmann.stocks.integration.dl4j;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.NoSuchElementException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.ta4j.core.Indicator;
import org.ta4j.core.num.Num;

import ch.pschatzmann.stocks.integration.HistoricValues;

/**
 * IStockRecordIterator using a 2d ND4j Array which is suited to be used by an
 * regular network
 * 
 */

public class StockData2DIterator implements DataSetIterator {
	private static final long serialVersionUID = 1L;
	private int windowSize;
	private int inputSize; // number of features for a stock data
	private int miniBatchSize; // mini-batch size
	private int outcomeSize; // default 1, say, one day ahead prediction
	private List<Integer> miniBatchOffsets = new ArrayList();
	private int recordCount = Integer.MAX_VALUE;	
	private DataSetPreProcessor preProcessor;
	private int dataSetStartIndex;
	private int dataSetEndIndex;
	private IteratorData data;

	/**
	 * Constructor with all parameters
	 * 
	 * @param id
	 * @param miniBatchSize
	 * @param exampleLength
	 * @param splitRatio
	 * @param category
	 */
	public StockData2DIterator(List<Indicator<Num>> inInd, List<Indicator<Num>> outInd, int batchSize, int windowSize) {
		this.data = new IteratorData(inInd, outInd);
		this.miniBatchSize = batchSize;		
		this.windowSize = windowSize;
		setupOffsets();
		this.outcomeSize = data.getOut().size();
		this.inputSize = data.getIn().size();
		data.getIn().forEach(v -> this.recordCount = Math.min(recordCount, v.size()));
		data.getOut().forEach(v -> this.recordCount = Math.min(recordCount, v.size()));
		
	}

	/**
	 * Constructor with no sliding window. 
	 * @param inInd
	 * @param outInd
	 * @param batchSize
	 */
	public StockData2DIterator(List<Indicator<Num>> inInd, List<Indicator<Num>> outInd, int batchSize) {
		this(inInd, outInd, batchSize, batchSize);
	}

	/**
	 * Initialize the mini-batch offsets. We use a sliding window for calculating
	 * the positions. If the window is equal the batch size there is no overlap. If
	 * the window is 1 we increment the starting position for each batch by 1
	 * 
	 **/
	protected void setupOffsets() {
		miniBatchOffsets.clear();
		int size = data.getIn().get(0).list().size();
		windowSize = Math.min(windowSize, size);
		
		for (int i = 0; i < size / windowSize; i++) {
			miniBatchOffsets.add(i * windowSize);
		}
	}


	
	@Override
	public DataSet next(int num) {
		if (miniBatchOffsets.size() == 0) {
			throw new NoSuchElementException();
		}

		dataSetStartIndex = miniBatchOffsets.remove(0);
		dataSetEndIndex = Math.min(dataSetStartIndex + this.miniBatchSize, this.recordCount-1);
		int actualMiniBatchSize = dataSetEndIndex - dataSetStartIndex;

		INDArray input = inputSize==0 ? null : Nd4j.create(new int[] { actualMiniBatchSize, inputSize }, 'f');	
		INDArray label = outcomeSize==0 ? null : Nd4j.create(new int[] { actualMiniBatchSize, outcomeSize }, 'f');

		for (int i = dataSetStartIndex; i < dataSetEndIndex; i++) {
			int column = i - dataSetStartIndex;

			int row = 0;
			for (HistoricValues historicValueFeature : data.getIn()) {
				input.putScalar(new int[] { column, row }, (historicValueFeature.list().get(i).getValue()));
				row++;
			}

			row = 0;
			for (HistoricValues historicValueLabel : data.getOut()) {
				label.putScalar(new int[] { column, row }, (historicValueLabel.list().get(i).getValue()));
				row++;
			}
		}

		DataSet result = new DataSet(input, label);
		if (this.getPreProcessor()!=null) {
			this.getPreProcessor().preProcess(result);
		}
		return result;
	}

	@Override
	public boolean hasNext() {
		return miniBatchOffsets.size() > 0;
	}
	
	@Override
	public int inputColumns() {
		return inputSize;
	}

	@Override
	public int totalOutcomes() {
		return outcomeSize;
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
		setupOffsets();
	}

	@Override
	public int batch() {
		return miniBatchSize;
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
		this.preProcessor = dataSetPreProcessor;
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		return this.preProcessor;
	}

	@Override
	public List<String> getLabels() {
		return data.getLabels();
	}

	@Override
	public DataSet next() {
		return next(miniBatchSize);
	}


	public int getRecordCount() {
		return this.recordCount;
	}
	
	/**
	 * Returns the start index of the current DataSet
	 * @return
	 */
	public int getDataSetStartIndex() {
		return this.dataSetStartIndex;
	}
	
	/**
	 * Returns the end index of the current DataSet
	 * @return
	 */
	public int getDataSetEndIndex() {
		return this.dataSetEndIndex;
	}
	
	public List<Date> getDates() {
		return this.data.getDates();
	}
	
	/**
	 * Writes the data as CSV to the outputstream
	 * @param os
	 * @throws IOException
	 */
	public void writeCSV(OutputStream os) throws IOException {
		this.data.writeCSV(os);
	}

}
