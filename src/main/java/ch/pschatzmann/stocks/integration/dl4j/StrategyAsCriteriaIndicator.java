package ch.pschatzmann.stocks.integration.dl4j;

import java.util.Arrays;
import java.util.List;

import org.ta4j.core.Indicator;
import org.ta4j.core.Strategy;
import org.ta4j.core.TimeSeries;
import org.ta4j.core.num.Num;

import ch.pschatzmann.stocks.Context;

/**
 * Translates a ta4j strategy into an indicator which can be used as outcome in
 * a StockRecordReader
 * 
 * @author pschatzmann
 *
 */
public class StrategyAsCriteriaIndicator implements Indicator<Num>, LabeledCategory {
	private static final long serialVersionUID = 1L;
	private Strategy strategy;
	private TimeSeries ts;
	private double defaultValue = 0;
	private boolean oneHot = true;

	public StrategyAsCriteriaIndicator(Strategy strategy, TimeSeries ts) {
		this.strategy = strategy;
		this.ts = ts;
	}

	/**
	 * sell => 0; buy => 1; hold=> 2
	 * 
	 * @param index
	 * @return
	 */
	@Override
	public Num getValue(int index) {
		try {
			boolean buy = strategy.getEntryRule().isSatisfied(index);
			boolean sell = strategy.getExitRule().isSatisfied(index);
			return numOf(buy ? 1.0 : sell ? 0.0 : 2.0);
		} catch(Exception ex) {
			// if there is an error we play it save and sell
			return numOf(defaultValue);
		}
	}

	@Override
	public TimeSeries getTimeSeries() {
		return ts;
	}

	public double getDefaultValue() {
		return defaultValue;
	}

	public void setDefaultValue(double defaultValue) {
		this.defaultValue = defaultValue;
	}

	@Override
	public List<String> getLabels() {
		return Arrays.asList("sell","buy","hold");
	}

	@Override
	public Num numOf(Number number) {
		return Context.number(number);
	}

	@Override
	public boolean isOneHotEncoded() {
		return oneHot;
	}
	
	public void setOneHotEncoded(boolean encoded) {
		this.oneHot = encoded;
	}

}
