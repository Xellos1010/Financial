import fetchAndSaveCandles from './coinbase/getProductCandles';
import { FibonacciBollingerBand } from './indicators/BollingerBands/fibonacciBollingerBand';
import { FibonacciBollingerBands } from './indicators/types/bollingerBands';
import { MovingAverageType } from './indicators/types/movingAverages';

async function executeStrategy() {
    const productId = 'BTC-USD';
    const granularity = '1800'; // Thirty minutes in seconds
    const startTime = Math.floor(Date.now() / 1000) - (7 * 24 * 60 * 60); // Last 7 days
    const endTime = Math.floor(Date.now() / 1000);

    const candles = await fetchAndSaveCandles(productId, granularity, startTime, endTime);
    const closePrices = candles.map((candle: any) => candle.close);
    const hlc3 = candles.map((candle: any) => (candle.high + candle.low + candle.close) / 3);

    const period = 200;
    const multiplier = 3.0;
    const maType: MovingAverageType = 'EMA'; // Example, you can change this

    const fibonacciBollingerBand = new FibonacciBollingerBand(closePrices, period, multiplier, hlc3, maType);
    const bands: FibonacciBollingerBands = fibonacciBollingerBand.calculate();

    console.log('Fibonacci Bollinger Bands:', bands);

    // Implement strategy logic
    // ...
}

executeStrategy().catch(console.error);
