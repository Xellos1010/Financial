// src/scripts/strategy/RSI_FibonacciBollingerBand.ts
import { BaseStrategy } from './baseStrategy';
import { RSI } from '../indicators/RSI/rsi';
import { FibonacciBollingerBand } from '../indicators/BollingerBands/fibonacciBollingerBand';
import { CoinbaseOrderExecution } from '../order_execution/coinbaseOrderExecution';
import { hlc3 } from '../utils';
import { MovingAverageType } from '../indicators/types/movingAverages';

interface Candle {
    time: number;
    low: number;
    high: number;
    open: number;
    close: number;
    volume: number;
}

export class RSI_FibonacciBollingerBand extends BaseStrategy<CoinbaseOrderExecution> {
    private rsiOverbought: number;
    private rsiOversold: number;
    private rsiLength: number;
    private fibonacciLength: number;
    private multiplier: number;

    constructor(productId: string, granularity: string, rsiOverbought: number, rsiOversold: number, rsiLength: number, fibonacciLength: number, multiplier: number, orderExecution: CoinbaseOrderExecution) {
        super(productId, granularity, orderExecution);
        this.rsiOverbought = rsiOverbought;
        this.rsiOversold = rsiOversold;
        this.rsiLength = rsiLength;
        this.fibonacciLength = fibonacciLength;
        this.multiplier = multiplier;
    }

    async initialize() {
        const endTime = Math.floor(Date.now() / 1000);
        const startTime = endTime - 300 * 60 * 30; // 300 bars of 30 minutes

        const candles: Candle[] = await this.fetchCandles(startTime, endTime);

        const prices = candles.map((candle: Candle) => candle.close);
        const hlc3Prices = candles.map((candle: Candle) => hlc3(candle));

        const rsiIndicator = new RSI(prices, this.rsiLength);
        const fibonacciBollingerBand = new FibonacciBollingerBand(prices, this.fibonacciLength, this.multiplier, hlc3Prices, 'sma' as MovingAverageType);
        

        this.addIndicator('RSI', rsiIndicator);
        this.addIndicator('FibonacciBollingerBand', fibonacciBollingerBand);

        this.generateSignals(candles);
    }

    generateSignals(candles: Candle[]) {
        const rsiOutput = this.indicators.find(indicator => indicator.name === 'RSI')!.instance.calculate().rsi;
        const fiboOutput = this.indicators.find(indicator => indicator.name === 'FibonacciBollingerBand')!.instance.calculate();

        for (let i = 0; i < candles.length; i++) {
            if (rsiOutput[i] > this.rsiOverbought) {
                console.log(`Sell Signal at ${candles[i].time}`);
                this.executeSellOrder(this.productId, candles[i].close);
            } else if (rsiOutput[i] < this.rsiOversold) {
                console.log(`Buy Signal at ${candles[i].time}`);
                this.executeBuyOrder(this.productId, candles[i].close);
            }
        }
    }
}
