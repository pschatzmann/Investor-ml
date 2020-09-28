package ch.pschatzmann.stocks.integration.dl4j;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.Serializable;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;

import org.ta4j.core.Indicator;
import org.ta4j.core.num.Num;

import ch.pschatzmann.stocks.accounting.HistoricValue;
import ch.pschatzmann.stocks.accounting.IHistoricValue;
import ch.pschatzmann.stocks.integration.HistoricValues;
import ch.pschatzmann.stocks.ta4j.indicator.Category;
import ch.pschatzmann.stocks.ta4j.indicator.IIndicator;

/**
 * Merges a list of input and ouput indicators
 * 
 * @author pschatzmann
 *
 */
public class IteratorData implements Serializable {
	private List<HistoricValues> in = new ArrayList();
	private List<HistoricValues> out = new ArrayList();
	private List<String> labels = new ArrayList();
	private int recordCount;
	private List<String> periods = null;
	private List<Date> dates;
	private String separatorCSV=",";
	
	public IteratorData(List<Indicator<Num>> inInd, List<Indicator<Num>> outInd) {
		// convert to historic values and build list set of all dates
		Collection<Date> allDates = new TreeSet();
		setupHistoricValues(inInd, in, false, allDates);
		setupHistoricValues(outInd, out, true, allDates);
		this.dates = new ArrayList(allDates);
		cleanupData(allDates);
	}
	
	public List<HistoricValues> getIn() {
		return this.in;
	}
	
	public List<HistoricValues> getOut() {
		return this.out;
	}
	
	public List<Date> getDates() {
		return dates;
	}
	
	public List<String> getLabels() {
		return this.labels;
	}
	
	protected void cleanupData(Collection<Date> allDates) {
		// find dates with invalid values
		Collection<Date> invalidDates = new TreeSet();
		//this.out.forEach(indicator -> collectInvalidDates(indicator, allDates, invalidDates));
		this.in.forEach(indicator -> collectInvalidDates(indicator, allDates, invalidDates));

		// remove invalid values
		for (Date date : invalidDates) {
			this.in.stream().forEach(historicValues -> historicValues.delete(date));
			this.out.stream().forEach(historicValues -> historicValues.delete(date));
			dates.remove(date);
		}

		for (HistoricValues i : this.in) {
			recordCount = Math.min(i.list().size(), recordCount);
		}
		for (HistoricValues i : this.out) {
			recordCount = Math.min(i.list().size(), recordCount);
		}
	}

	protected void setupHistoricValues(List<Indicator<Num>> indicators, List<HistoricValues> valuesList, boolean labels,
			Collection<Date> dates) {
		indicators.stream().forEach(indicator -> addHistoricValue(indicator, valuesList, labels, dates));
	}

	protected void addHistoricValue(Indicator<Num> indicator, List<HistoricValues> valuesList, boolean labels,
			Collection<Date> dates) {
		HistoricValues historicValues = HistoricValues.create(indicator);
		dates.addAll(historicValues.getDistinctDates());
		if (indicator instanceof Category && ((Category) indicator).isOneHotEncoded()) {
			Set<Long> codes = getAvailableCodes(historicValues);
			for (Long v : codes) {
				valuesList.add(oneHotEncoded(historicValues, v));
				if (labels) {
					this.labels.add(historicValues.getName() + "-" + v);
				}
			}
		} else {
			valuesList.add(historicValues);
			if (labels) {
				this.labels.add(historicValues.getName());
			}
		}
	}

	protected Set<Long> getAvailableCodes(HistoricValues historicValues) {
		Set<Long> codes = new TreeSet();
		for (IHistoricValue hv : historicValues.list()) {
			Double value = hv.getValue();
			if (value != null) {
				codes.add(value.longValue());
			}
		}
		return codes;
	}

	protected HistoricValues oneHotEncoded(HistoricValues values, Long value) {
		List<IHistoricValue> result = new ArrayList();
		for (IHistoricValue hv : values.list()) {
			Long newValue = value.equals(hv.getValue().longValue()) ? 1l : 0l;
			result.add(new HistoricValue(hv.getDate(), newValue.doubleValue()));
		}
		return HistoricValues.create(result, "" + value);
	}

	protected void collectInvalidDates(HistoricValues historicValues, Collection<Date> dates,
			Collection<Date> invalidDates) {
		invalidDates.addAll(historicValues.getInvalidDates(dates));

	}
	
	/**
	 * Returns the number of data records
	 * @return
	 */
	public int getSize() {
		return out.isEmpty()? in.get(0).size() : Math.max(out.get(0).size(),in.get(0).size());
	}
	
	
	public void writeCSV(OutputStream os) throws IOException {
		BufferedOutputStream bos = new BufferedOutputStream(os);
		String header = getHeaderCSV();
		bos.write(header.getBytes());
		for (Date date : this.getDates()) {
			String line = getLineCSV(date);
			bos.write(line.getBytes());
		}
		bos.flush();
	}

	private String getHeaderCSV() {
		StringBuffer sb = new StringBuffer();
		for (HistoricValues hv : this.in) {
			if (sb.length()>0) {
				sb.append(separatorCSV);
			}
			sb.append(hv.getName());		
		}
		for (HistoricValues hv : this.out) {
			if (sb.length()>0) {
				sb.append(separatorCSV);
			}
			sb.append(hv.getName());					
		}
		sb.append(System.lineSeparator());
		return sb.toString();
	}

	private String getLineCSV(Date date) {
		StringBuffer sb = new StringBuffer();
		for (HistoricValues hv : this.in) {
			if (sb.length()>0) {
				sb.append(separatorCSV);
			}
			sb.append(hv.getValue(date));		
		}
		for (HistoricValues hv : this.out) {
			if (sb.length()>0) {
				sb.append(separatorCSV);
			}
			sb.append(hv.getValue(date));		
		}
		sb.append(System.lineSeparator());
		return sb.toString();
	}

	public String getSeparatorCSV() {
		return separatorCSV;
	}

	public void setSeparatorCSV(String separatorCSV) {
		this.separatorCSV = separatorCSV;
	}
	
	

}
