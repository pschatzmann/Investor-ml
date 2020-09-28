package ch.pschatzmann.stocks.integration.dl4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import ch.pschatzmann.dates.CalendarUtils;
import ch.pschatzmann.stocks.accounting.HistoricValue;
import ch.pschatzmann.stocks.accounting.IHistoricValue;
import ch.pschatzmann.stocks.forecasting.BaseForecast;
import ch.pschatzmann.stocks.integration.HistoricValues;

/**
 * Generates a forecast using a DL4J LSTM RNN network. Please note that we
 * provide only 1 forecast value
 * 
 * @author pschatzmann
 *
 */
public class LSTMForecast extends BaseForecast {
	private static final long serialVersionUID = 1L;
	private MultiLayerNetwork net;
	private StockData3DIterator it;
	private RegressionEvaluation ev;
	private boolean onlyPredictions = false;
	private int epochs = 0;
	private HistoricValues historicValues = null;

	public LSTMForecast(StockData3DIterator it, int epochs) {
		this(it, getDefaultMultiLayerConfiguration(it), epochs);
	}

	public LSTMForecast(StockData3DIterator it, MultiLayerConfiguration conf, int epochs) {
		this.it = it;
		setDefaultName();
		net = new MultiLayerNetwork(conf);
		net.init();

		this.epochs = epochs;
		train();
	}

	public LSTMForecast(StockData3DIterator it, MultiLayerNetwork trainedNetwork) {
		this.it = it;
		this.net = trainedNetwork;
		setDefaultName();
	}

	protected static MultiLayerConfiguration getDefaultMultiLayerConfiguration(StockData3DIterator it) {
		int periods = it.inputPeriods() + it.outcomePeriods();
		int lstmLayer1Size = periods * 2;
		int lstmLayer2Size = periods;
		int truncatedBPTTLength = 250;
		double dropoutRatio = 0.8;
		int nIn = it.inputColumns();
		int nOut = it.totalOutcomes();

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(1234)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).weightInit(WeightInit.XAVIER)
				.updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD) // RMSPROP or ADAGRAD
				.l2(1e-2).list()
				.layer(0,
						new LSTM.Builder().nIn(nIn).nOut(lstmLayer1Size).gateActivationFunction(Activation.SOFTSIGN)
								.dropOut(dropoutRatio).build())
				.layer(1,
						new LSTM.Builder().nIn(lstmLayer1Size).nOut(lstmLayer2Size)
								.gateActivationFunction(Activation.SOFTSIGN).dropOut(dropoutRatio).build())
				.layer(2,
						new RnnOutputLayer.Builder().nIn(lstmLayer2Size).nOut(nOut).activation(Activation.IDENTITY)
								.lossFunction(LossFunctions.LossFunction.MSE).build())
				.backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(truncatedBPTTLength)
				.tBPTTBackwardLength(truncatedBPTTLength).pretrain(false).backprop(true).build();
		return conf;
	}

	protected void setDefaultName() {
		this.setName(this.getClass().getSimpleName());
	}

	protected void train() {
		for (int i = 0; i < epochs; i++) {
			while (it.hasNext()) {
				net.fit(it.next());
			}
		}
		epochs = 0;
	}

	/**
	 * The number of forecasts is ignored because we support only 1 forecast
	 */

	@Override
	public HistoricValues forecast(int numberOfForecasts) throws Exception {
		if (this.historicValues == null) {
			ev = new RegressionEvaluation(it.totalOutcomes());
			List<IHistoricValue> result = new ArrayList();

			List<Date> dates = it.getDates();
			Date lastDate = dates.get(dates.size() - 1);
			Date nextDate = CalendarUtils.nextWorkDay(lastDate);

			List<Date> allDates = new ArrayList(dates);
			allDates.add(nextDate);

			it.reset();
			while (it.hasNext()) {
				ScalingDataSet data = (ScalingDataSet) it.next();
				INDArray mask = data.getLabelsMaskArray();
				INDArray labels = data.getLabels();
				data.revertLabels(labels, mask);

				INDArray prediction = net.rnnTimeStep(data.getFeatures());
				data.revertLabels(prediction);

				ev.eval(labels, prediction);
				for (int j = 0; j < it.inputPeriods(); j++) {				
					if (mask.getDouble(0l, j) == 1.0) {
						// we have labels and predictions
						if (allDates.size() == 1 || isOnlyPredictions()) {
							result.add(new HistoricValue(allDates.remove(0), prediction.getDouble(0l, 0l, j)));
						} else {
							result.add(new HistoricValue(allDates.remove(0), labels.getDouble(0l, 0l, j)));
						}
					} else {
						// we have only a prediction
						if (data.getFeaturesMaskArray().getDouble(0, j) == 1.0) {
							result.add(new HistoricValue(allDates.remove(0), prediction.getDouble(0l, 0l, j)));
						}
					}
				}
			}
			
			this.historicValues = HistoricValues.create(result, this.getName());
		}

		return this.historicValues;
	}

	/**
	 * we provide only predictions instead of labels
	 * 
	 * @return
	 */

	public boolean isOnlyPredictions() {
		return onlyPredictions;
	}

	/**
	 * If set to true we provide only predictions. If set to false we provide labels
	 * if possible and only a prediction if there is not label
	 * 
	 * @param onlyPredictions
	 */
	public void setOnlyPredictions(boolean onlyPredictions) {
		this.onlyPredictions = onlyPredictions;
	}

	/**
	 * Provides the evaluations.
	 * 
	 * @return
	 * @throws Exception
	 */
	public RegressionEvaluation getRegressionEvaluation() throws Exception {
		if (ev == null) {
			if (epochs > 0)
				train();
			forecast(1);
		}
		return ev;
	}

	/**
	 * Returns the trained network
	 * 
	 * @return
	 */
	public MultiLayerNetwork getNet() {
		if (epochs > 0)
			train();
		return this.net;
	}

	/**
	 * Loads the network
	 * 
	 * @param file
	 * @return
	 * @throws IOException
	 */
	public MultiLayerNetwork load(File file) throws IOException {
		return ModelSerializer.restoreMultiLayerNetwork(file);

	}

	/**
	 * Saves the network
	 * 
	 * @param file
	 * @throws IOException
	 */
	public void save(File file) throws IOException {
		ModelSerializer.writeModel(net, file, true);
	}

}
