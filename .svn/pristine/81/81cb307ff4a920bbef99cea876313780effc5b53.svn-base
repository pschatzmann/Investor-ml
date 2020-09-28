package ch.pschatzmann.evaluation;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Confusion Matrix and related KPIs. We can add the labels directly as Strings
 * or their corresponding numeric values. In the later case we migth privde the corresponding
 * String values by calling setLabelTitles()
 * 
 * @author pschatzmann
 *
 */
public class ConfusionMatrix<T> {
	private final Map<T, Map<T, Long>> linearizedMatrix = new TreeMap();
	private Map<T, String> labelTitles = new TreeMap();
	private long numberOfEvaluatedDocs = 0;
	private double accuracy = -1d;
	private String titleField = " ";

	/**
	 * Empty Constructor
	 */
	public ConfusionMatrix() {
	}

	/**
	 * Constructor with linearizedMatrix
	 * 
	 * @param map
	 */
	public ConfusionMatrix(Map<T, Map<T, Long>> map) {
		add(map);
	}

	/**
	 * Constructor which creates a copy of an existing ConfusionMatrix
	 * 
	 * @param matrix
	 */
	public ConfusionMatrix(ConfusionMatrix<T> matrix) {
		add(matrix.getLinearizedMatrix());
	}

	/**
	 * Adds 1 label
	 * 
	 * @param currentLabel
	 * @param correctLabel
	 */
	public void add(T correctLabel, T predictedLabel) {
		add(correctLabel, predictedLabel, 1l);
	}

	/**
	 * Adds count labels
	 * 
	 * @param currentLabel
	 * @param correctLabel
	 * @param count
	 */
	public synchronized void add(T correctLabel, T predictedLabel, long count) {
		numberOfEvaluatedDocs += count;
		Map<T, Long> stringLongMap = linearizedMatrix.get(correctLabel);
		if (stringLongMap != null) {
			Long aLong = stringLongMap.get(predictedLabel);
			if (aLong != null) {
				stringLongMap.put(predictedLabel, aLong + count);
			} else {
				stringLongMap.put(predictedLabel, count);
			}
		} else {
			stringLongMap = new HashMap<>();
			stringLongMap.put(predictedLabel, count);
			linearizedMatrix.put(correctLabel, stringLongMap);
		}
	}

	/**
	 * Adds all values of a linearizedMatrix
	 * 
	 * @param map
	 */
	public synchronized void add(Map<T, Map<T, Long>> map) {
		for (Entry<T, Map<T, Long>> e : map.entrySet()) {
			for (Entry<T, Long> e1 : e.getValue().entrySet()) {
				add(e1.getKey(), e.getKey(), e1.getValue());
			}
		}
	}
	
	/**
	 * Adds all values of a ConfusionMatrix
	 * @param matrix
	 */
	public void add(ConfusionMatrix<T> matrix) {
		add(this.getLinearizedMatrix());
	}

	/**
	 * get the linearized confusion matrix as a {@link Map}
	 *
	 * @return a {@link Map} whose keys are the correct classification answers and
	 *         whose values are the actual answers' counts
	 */
	public Map<T, Map<T, Long>> getLinearizedMatrix() {
		return Collections.unmodifiableMap(linearizedMatrix);
	}

	/**
	 * @return Content of first table title field on label title column
	 */
	public String getTitleField() {
		return titleField;
	}

	/**
	 * Defines the Content of first table title field on the label title column
	 * 
	 * @param titleField
	 */
	public void setTitleField(String titleField) {
		this.titleField = titleField;
	}

	public Map<T, String> getLabelTitles() {
		return labelTitles;
	}

	public void setLabelTitles(Map<T, String> labelTitles) {
		this.labelTitles = labelTitles;
	}


	/**
	 * Calculate accuracy on this confusion matrix 
	 * 
	 * @return the accuracy
	 */
	public double getAccuracy() {
		if (this.accuracy == -1.0) {
			double correct = 0.0;
			for (Map.Entry<T, Map<T, Long>> classification : linearizedMatrix.entrySet()) {
				T label = classification.getKey();
				for (Map.Entry<T, Long> entry : classification.getValue().entrySet()) {
					if (label.equals(entry.getKey())) {
						correct += entry.getValue();
					}
				}
			}
			this.accuracy = correct / this.size();
		}
		return this.accuracy;
	}



	/**
	 * get the no. of documents evaluated while generating this confusion matrix
	 *
	 * @return the no. of documents evaluated
	 */
	public long getNumberOfEvaluatedDocs() {
		return numberOfEvaluatedDocs;
	}

	/**
	 * Provides all labels
	 * 
	 * @return Set of all labels
	 */
	public List<T> getLabels() {
		Set<T> result = new TreeSet();
		for (Entry<T, Map<T, Long>> e : linearizedMatrix.entrySet()) {
			for (Entry<T, Long> e1 : e.getValue().entrySet()) {
				result.add(e.getKey());
				result.add(e.getKey());
			}
		}
		return new ArrayList(result);
	}

	/**
	 * Provides the confusion matrix as html table
	 * 
	 * @return String with html table
	 */
	public String html() {
		StringBuffer sb = new StringBuffer();
		sb.append("<table class='ConfusionMatrixTable'>");
		sb.append("<tr class='ConfusionMatrixTR0'>");
		sb.append("<th class='ConfusionMatrixTH0'></th>");
		sb.append("<th class='ConfusionMatrixTH0'></th>");
		sb.append("<th colspan='"+this.getLabels().size()+"' class='ConfusionMatrixTH0'>");
		sb.append("Predicted");
		sb.append("</th>");		
		sb.append("<tr class='ConfusionMatrixTR'>");
		sb.append("<th class='ConfusionMatrixTH'></th>");
		sb.append("<th class='ConfusionMatrixTH'></th>");
		for (T fld : getLabels()) {
			sb.append("<th class='ConfusionMatrixTH'>");
			sb.append(getTitle(fld));
			sb.append("</th>");
		}
		sb.append("</tr>");

		sb.append("<tr class='ConfusionMatrixTR0'>");
	    sb.append("<th class='ConfusionMatrixTH0' rowspan='"+this.getLabels().size()+1+"'>Actual</th>");
		sb.append("</tr>");
		
		for (T fld : getLabels()) {
			sb.append("<tr class='ConfusionMatrixTR'>");
			sb.append("<th class='ConfusionMatrixTH1'>");
			sb.append(getTitle(fld));
			sb.append("</th>");

			for (T fldPredicted : getLabels()) {
				sb.append("<td class='ConfusionMatrixTD'>");
				sb.append(String.valueOf(getClassificationCount(fld, fldPredicted)));
				sb.append("</td>");
			}
			sb.append("</tr>");
		}
		sb.append("</table>");
		return sb.toString();
	}

	/**
	 * Provides the confusion matrix as csv String
	 * 
	 * @return csv String
	 */
	public String csv() {
		StringBuffer sb = new StringBuffer();
		sb.append(this.getTitleField());
		for (T fld : getLabels()) {
			sb.append(",");
			sb.append(getTitle(fld));
		}
		sb.append(System.lineSeparator());

		for (T fld : getLabels()) {
			sb.append(getTitle(fld));
			for (T fldPredicted : getLabels()) {
				sb.append(",");
				sb.append(String.valueOf(getClassificationCount(fld, fldPredicted)));
			}
			sb.append(System.lineSeparator());
		}
		return sb.toString();
	}

	/**
	 * Provides the confusion matrix as list of Map records
	 * 
	 * @return List of records where the record is represented as Map
	 */
	public List<Map<T, Object>> list() {
		List<Map<T, Object>> result = new ArrayList();
		for (T fld : getLabels()) {
			Map record = new TreeMap();
			record.put(this.getTitleField(), fld);
			for (T fldPredicted : getLabels()) {
				record.put(getTitle(fldPredicted), String.valueOf(getClassificationCount(fld, fldPredicted)));
			}
			result.add(record);
		}
		return result;
	}

	protected String getTitle(T label) {
		String result = this.labelTitles.get(label);
		return result == null ? String.valueOf(label) : result;
	}

	/**
	 * Provides the count for the indicated label combination
	 * 
	 * @param currentLabel
	 * @param correctLabel
	 * @return
	 */
	public long getClassificationCount(T correctLabel, T predictedLabel) {
		Long result = 0l;
		Map<T, Long> map = linearizedMatrix.get(correctLabel);
		if (map != null) {
			result = map.get(predictedLabel);
			if (result == null) {
				result = 0l;
			}
		}
		return result;
	}

	/**
	 * Clones this ConfusionMatrix
	 */
	public ConfusionMatrix clone() {
		return new ConfusionMatrix(this);
	}

	/**
	 * 
	 * @return number of evaluated documents
	 */
	public long size() {
		return this.numberOfEvaluatedDocs;
	}

}
