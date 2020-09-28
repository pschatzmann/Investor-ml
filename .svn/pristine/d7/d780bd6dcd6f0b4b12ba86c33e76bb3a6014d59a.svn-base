package ch.pschatzmann.stocks.integration.dl4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.AbstractDataSetNormalizer;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;

/**
 * Scaling for each dataset 
 * 
 * @author pschatzmann
 *
 */

public class ScalingDataSet extends DataSet {
	private static final long serialVersionUID = 1L;
	private NormalizerMinMaxScalerLabelsFromFeatures normalizer;

	public ScalingDataSet(INDArray in, INDArray out, INDArray in1, INDArray out1, NormalizerMinMaxScalerLabelsFromFeatures norm){
		super(in, out, in1, out1);
		this.normalizer = norm;
		normalizer.fit(this);
		if (normalizer.isFit()) {
			normalizer.transform(this);
		} 
	}
	
	public void revert() {
		if (normalizer!=null)
			normalizer.revert(this);
	}
	
	public void revertFeatures(INDArray features) {
		if (normalizer!=null)
			normalizer.revertFeatures(features);
	}

	public void revertLabels(INDArray labels) {
		if (normalizer!=null)
			normalizer.revertLabels(labels);
	}
	
	public void revertFeatures(INDArray features, INDArray mask) {
		if (normalizer!=null)
			normalizer.revertFeatures(features, mask);
	}

	public void revertLabels(INDArray labels, INDArray mask) {
		if (normalizer!=null)
			normalizer.revertLabels(labels, mask);
	}
		
}
