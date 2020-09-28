package ch.pschatzmann.stocks.integration.dl4j;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.AbstractDataSetNormalizer;
import org.ta4j.core.BaseStrategy;
import org.ta4j.core.Indicator;
import org.ta4j.core.Rule;
import org.ta4j.core.Strategy;
import org.ta4j.core.TradingRecord;
import org.ta4j.core.num.Num;

/**
 * Strategy which is driven by a DL4J Model. The model must be set up so that the first outcome
 * column (0) contains the sell signals and the second (1) contains the buy signals. Any other
 * column will be ignored
 * 
 * @author pschatzmann
 *
 */
public class DL4JStrategy implements Strategy {
	private int unstablePeriod;
	private List<Boolean> buyList = new ArrayList();
	private List<Boolean> sellList = new ArrayList();

	// inline buy rule
	private Rule entryRule = new Rule() {
		@Override
		public boolean isSatisfied(int index, TradingRecord tradingRecord) {
			return buyList.get(index);
		}
	};

	// inline sell rule
	private Rule exitRule = new Rule() {
		@Override
		public boolean isSatisfied(int index, TradingRecord tradingRecord) {
			return sellList.get(index);
		}
	};

	/**
	 * Constructor
	 * @param model
	 * @param in
	 * @param batchSize
	 * @param normalizer
	 */
	
	public DL4JStrategy(MultiLayerNetwork model, List<Indicator<Num>> in, int batchSize,
			AbstractDataSetNormalizer normalizer) {

		StockData2DIterator iterator = new StockData2DIterator(in, new ArrayList(), batchSize);
		while (iterator.hasNext()) {
			DataSet allData = iterator.next();
			if (normalizer!=null) {
				normalizer.fit(allData);
				normalizer.transform(allData);
			}
			INDArray predicted = model.output(allData.getFeatures(), false);
			INDArray sell = predicted.getColumn(0);
			for (double v : sell.data().asDouble()) {
				sellList.add(v > 0.99);
			}

			INDArray buy = predicted.getColumn(1);
			for (double v : buy.data().asDouble()) {
				buyList.add(v > 0.99);
			}
		}
	}

	@Override
	public Rule getEntryRule() {
		return entryRule;
	}

	@Override
	public Rule getExitRule() {
		return exitRule;
	}

	@Override
	public void setUnstablePeriod(int unstablePeriod) {
		this.unstablePeriod = unstablePeriod;
	}

	@Override
	public boolean isUnstableAt(int index) {
		return index < unstablePeriod;
	}

    @Override
    public Strategy and(Strategy strategy) {
        String andName = "and(" + this.getName() + "," + strategy.getName() + ")";
        int unstable = Math.max(unstablePeriod, strategy.getUnstablePeriod());
        return and(andName, strategy, unstable);
    }

    @Override
    public Strategy or(Strategy strategy) {
        String orName = "or(" + this.getName() + "," + strategy.getName() + ")";
        int unstable = Math.max(unstablePeriod, strategy.getUnstablePeriod());
        return or(orName, strategy, unstable);
    }

    @Override
    public Strategy opposite() {
        return new BaseStrategy("opposite(" + getName() + ")", exitRule, entryRule, unstablePeriod);
    }

    @Override
    public Strategy and(String name, Strategy strategy, int unstablePeriod) {
        return new BaseStrategy(name, entryRule.and(strategy.getEntryRule()), exitRule.and(strategy.getExitRule()), unstablePeriod);
    }

    @Override
    public Strategy or(String name, Strategy strategy, int unstablePeriod) {
        return new BaseStrategy(name, entryRule.or(strategy.getEntryRule()), exitRule.or(strategy.getExitRule()), unstablePeriod);
    }

	@Override
	public String getName() {
		return this.getClass().getSimpleName();
	}

	@Override
	public int getUnstablePeriod() {
		return unstablePeriod;
	}


}
