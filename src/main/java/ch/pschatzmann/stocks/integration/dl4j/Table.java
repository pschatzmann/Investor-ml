package ch.pschatzmann.stocks.integration.dl4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Table which supports the display of a 2d dataset
 * 
 * @author pschatzmann
 *
 */
public class Table extends ch.pschatzmann.display.Table {
	protected Table(List<Map> listOfMap) {
		super(listOfMap);
	}

	public static Table create(DataSet ds) {
		INDArray f = ds.getFeatures();
		INDArray l = ds.getLabels();

		List labelNames = ds.getLabelNamesList();
		if (labelNames == null || labelNames.isEmpty()) {
			labelNames = new ArrayList();
			for (int col = 0; col < l.columns(); col++) {
				labelNames.add("label-" + col);
			}
		}

		List result = new ArrayList();
		for (int row = 0; row < f.rows(); row++) {
			Map rec = new TreeMap();
			for (int col = 0; col < f.columns(); col++) {
				rec.put("feature-" + col, f.getDouble(row, col));
			}
			for (int col = 0; col < l.columns(); col++) {
				rec.put(labelNames.get(col), l.getDouble(row, col));
			}
			result.add(rec);
		}
		return new Table(result);

	}
}
