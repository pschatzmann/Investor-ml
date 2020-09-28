package ch.pschatzmann.stocks.integration.dl4j;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetUtil;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.stats.MinMaxStats;
import org.nd4j.linalg.factory.Nd4j;

/**
 * NormalizerMinMaxScaler where the feature statistics are applied to the labels
 * as well
 * 
 * @author pschatzmann
 *
 */

public class NormalizerMinMaxScalerLabelsFromFeatures extends NormalizerMinMaxScaler {
	private static final long serialVersionUID = 1L;
	private List<Long>positions = null;
	
	public NormalizerMinMaxScalerLabelsFromFeatures(List<Long>positions) {
		this.positions = positions;
	}
	
	
	@Override
	public void fit(DataSet dataSet) {
        if (DataSetUtil.tailor2d(dataSet.getFeatures(), dataSet.getFeaturesMaskArray())!=null) {
		   MinMaxStats stats = (MinMaxStats) newBuilder().addFeatures(dataSet).build();
		   setFeatureStats(stats);
		   if (isFitLabel() && positions!=null) {
			   INDArray min = Nd4j.zeros(1,positions.size(),'f');
			   INDArray max = Nd4j.zeros(1,positions.size(),'f');
			   for (long pos : positions) {
				   min.putScalar(0l,pos, stats.getLower().getDouble(0l, pos));
				   max.putScalar(0l,pos, stats.getUpper().getDouble(0l, pos));			
				}
				setLabelStats(new MinMaxStats(min,max));
			}
		}
	}
	
	public boolean isFit() {
		return super.isFit();
	}
	
    public void transform(INDArray features, INDArray featuresMask) {
        if (getFeatureStats()!=null) {
        	super.transform(features, featuresMask);
        }
    }
    
    public void revertLabels(INDArray labels, INDArray labelsMask) {
        if (getFeatureStats()!=null) {
        	super.revertLabels(labels, labelsMask);
        }
    }

    @Override
    public void revertFeatures(INDArray features, INDArray featuresMask) {
        if (getFeatureStats()!=null) {
        	super.revertFeatures(features, featuresMask);
        }
    }

	
    public MinMaxStats getFeatureStats() {
        return super.getFeatureStats();
    }

    protected MinMaxStats getLabelStats() {
        return super.getLabelStats();
    }


}
