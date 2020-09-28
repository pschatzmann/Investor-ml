package ch.pschatzmann.stocks.integration.dl4j;

import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

import org.ta4j.core.Bar;
import org.ta4j.core.Indicator;
import org.ta4j.core.num.Num;

import ch.pschatzmann.stocks.ta4j.indicator.SplitIndicator;

/**
 * Splits Indicators into a training and test set
 * 
 * @author pschatzmann
 *
 */

public class IndicatorSplitter {
	
	/**
	 * Provides the head (training data) or tail (test data) as indicated by the head parameter
	 * @param lists
	 * @param factor
	 * @param head
	 * @return
	 */
	public static List<Indicator<Num>> split(List<Indicator<Num>> lists, Double factor, boolean head) {
		return lists.stream().map(indicator -> new SplitIndicator(indicator, factor, head))
				.collect(Collectors.toList());
	}

	/**
	 * Provides the head (training data) or tail (test data) as indicated by the head parameter
	 * @param lists
	 * @param factor
	 * @param head
	 * @return
	 */
	public static List<Indicator<Num>> split(List<Indicator<Num>> lists, Date date, boolean head) {
		int pos = getSplitPos(lists,date);
		return split(lists, pos, head);
	}

	
	/**
	 * Provides the head (training data) or tail (test data) as indicated by the head parameter
	 * @param lists
	 * @param pos
	 * @param head
	 * @return
	 */
	public static List<Indicator<Num>> split(List<Indicator<Num>> lists, Integer pos, boolean head) {
		return lists.stream().map(indicator -> new SplitIndicator(indicator, pos, head))
				.collect(Collectors.toList());
	}
	

	/**
	 * Returns the split position
	 * @param lists
	 * @param date
	 * @param offset
	 * @return
	 */
	public static Integer getSplitPos(List<Indicator<Num>> lists, Date date) {
		return getSplitPos(lists,date,0);
	}

	/**
	 * Returns the split position
	 * @param lists
	 * @param date
	 * @return
	 */
	public static Integer getSplitPos(List<Indicator<Num>> lists, Date date, Integer offset) {
		List<Bar> barData = lists.get(0).getTimeSeries().getBarData();
		int pos = 0;
		for (Bar bar :barData) {
			if (Date.from(bar.getBeginTime().toInstant()).after(date)){
				break;
			}
			pos++;
		}
		return Math.min(Math.max(0, pos+offset), barData.size()-1);		
	}

}
