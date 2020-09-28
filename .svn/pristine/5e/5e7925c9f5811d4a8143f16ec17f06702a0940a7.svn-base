package ch.pschatzmann.investor.tests;


import java.util.Arrays;

import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.ta4j.core.Indicator;
import org.ta4j.core.TimeSeries;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.ta4j.core.indicators.helpers.MaxPriceIndicator;
import org.ta4j.core.indicators.helpers.MinPriceIndicator;
import org.ta4j.core.indicators.helpers.OpenPriceIndicator;
import org.ta4j.core.num.Num;

import ch.pschatzmann.stocks.Context;
import ch.pschatzmann.stocks.IStockData;
import ch.pschatzmann.stocks.input.MarketArchiveHttpReader;
import ch.pschatzmann.stocks.integration.StockTimeSeries;
import ch.pschatzmann.stocks.integration.dl4j.StockData2DIterator;
import ch.pschatzmann.stocks.integration.dl4j.StockData3DIterator;
import ch.pschatzmann.stocks.ta4j.indicator.FunctionalCategoryIndicator;
import ch.pschatzmann.stocks.ta4j.indicator.OffsetIndicator;
import ch.pschatzmann.stocks.ta4j.indicator.PercentChangeIndicator;

public class TestDl4j {
	
	@BeforeClass
	public static void setup() throws Exception {
		System.out.println("*** "+TestDl4j.class.getSimpleName()+" ***");
		Context.setDefaultReader(new MarketArchiveHttpReader());
	}

	

	@Test
	public void testReaderRead() throws Exception {
		IStockData sd = Context.getStockData("AAPL","NASDAQ");
		Indicator closePrice = new ClosePriceIndicator(new StockTimeSeries(sd));
		Indicator closePrice1 = new OffsetIndicator(closePrice, -1);
		Indicator percentChange = new PercentChangeIndicator(closePrice);

		Indicator p1 = new OffsetIndicator(percentChange, -1);
		Indicator p2 = new OffsetIndicator(percentChange, -2);
		Indicator p3 = new OffsetIndicator(percentChange, -3);
		Indicator p4 = new OffsetIndicator(percentChange, -4);
		Indicator p5 = new OffsetIndicator(percentChange, -5);
		Indicator result = new FunctionalCategoryIndicator(closePrice1, closePrice, (v1, v2) -> (v2 > v1) ? 1 : 0);

//		StockRecordReader recordReader = new StockRecordReader(Arrays.asList(p1, p2, p3, p4, p5),
//				Arrays.asList(result));
//		RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 10000);
//		System.out.println(iterator.getLabels());
//		DataSet allData = iterator.next();
//		SplitTestAndTrain split = allData.splitTestAndTrain(0.9);
//		DataSet train = split.getTrain();
//		DataSet test = split.getTest();
//		System.out.println(test);
		
		DataSetIterator iterator = new StockData2DIterator(Arrays.asList(p1, p2, p3, p4, p5),
				Arrays.asList(result), 10000);
		DataSet ds = iterator.next();
		System.out.println(ds);


	}

	@Test
	public void testDataSetIterator() throws Exception {
		IStockData sd = Context.getStockData("AAPL","NASDAQ");
		Indicator closePrice = new ClosePriceIndicator(new StockTimeSeries(sd));
		Indicator closePrice1 = new OffsetIndicator(closePrice, -1);
		Indicator percentChange = new PercentChangeIndicator(closePrice);

		Indicator p1 = new OffsetIndicator(percentChange, -1);
		Indicator p2 = new OffsetIndicator(percentChange, -2);
		Indicator p3 = new OffsetIndicator(percentChange, -3);
		Indicator p4 = new OffsetIndicator(percentChange, -4);
		Indicator p5 = new OffsetIndicator(percentChange, -5);
		Indicator result = new FunctionalCategoryIndicator(closePrice1, closePrice, (v1, v2) -> (v2 > v1) ? 1.0 : 0.0);

		DataSetIterator iterator = new StockData2DIterator(Arrays.asList(p1, p2, p3, p4, p5), Arrays.asList(result),
				10000);
		System.out.println(iterator.getLabels());
		DataSet allData = iterator.next();
		SplitTestAndTrain split = allData.splitTestAndTrain(0.9);
		DataSet train = split.getTrain();
		DataSet test = split.getTest();
		System.out.println(test);

	}

	@Test
	public void test2dAll() {
		IStockData sd = Context.getStockData("AAPL","NASDAQ");
		TimeSeries timeSeries = new StockTimeSeries(sd);
		Indicator<Num> close = new ClosePriceIndicator(timeSeries);
		Indicator<Num> open = new OpenPriceIndicator(timeSeries);
		Indicator<Num> high = new MaxPriceIndicator(timeSeries);
		Indicator<Num> low = new MinPriceIndicator(timeSeries);
		Indicator<Num> volume = new ch.pschatzmann.stocks.ta4j.indicator.VolumeIndicator(timeSeries);
		Indicator<Num> closePredicted = new OffsetIndicator(close,+1);

		DataSetIterator iterator = new StockData2DIterator(Arrays.asList(close, open, high, low, volume), Arrays.asList(closePredicted), 100);
		DataSet ds = iterator.next();
		System.out.println(ds);

	}
	

	@Test
	public void test3dAll() {
		IStockData sd = Context.getStockData("AAPL","NASDAQ");
		TimeSeries timeSeries = new StockTimeSeries(sd);
		Indicator<Num> close = new ClosePriceIndicator(timeSeries);
		Indicator<Num> open = new OpenPriceIndicator(timeSeries);
		Indicator<Num> high = new MaxPriceIndicator(timeSeries);
		Indicator<Num> low = new MinPriceIndicator(timeSeries);
		Indicator<Num> volume = new ch.pschatzmann.stocks.ta4j.indicator.VolumeIndicator(timeSeries);

		DataSetIterator iterator = new StockData3DIterator(Arrays.asList(close, open, high, low, volume), Arrays.asList(close));
		DataSet ds = iterator.next();		
		System.out.println(ds);
		ds = iterator.next();		
		System.out.println(ds);

	}
	
	
	
	@Test
	public void testSequence() throws Exception {
		IStockData sd = Context.getStockData("AAPL","NASDAQ");
		Indicator closePrice = new ClosePriceIndicator(new StockTimeSeries(sd));
		
		DataSetIterator iterator = new StockData3DIterator(Arrays.asList(closePrice),
				Arrays.asList(closePrice));
		DataSet ds = iterator.next();
		System.out.println(ds);


	}

	
}
