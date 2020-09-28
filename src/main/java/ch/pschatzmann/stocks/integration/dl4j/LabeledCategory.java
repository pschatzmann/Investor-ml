package ch.pschatzmann.stocks.integration.dl4j;

import java.util.List;

import ch.pschatzmann.stocks.ta4j.indicator.Category;

/**
 * Category for which we provide the corresponding labels
 * 
 * @author pschatzmann
 *
 */
public interface LabeledCategory extends Category {
	List<String> getLabels();
}
