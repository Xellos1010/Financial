// src/scripts/indicators/MovingAverages/vwma.ts
import { MovingAverageBase } from './movingAverageBase';
import { MovingAverageOutput } from '../types/movingAverages';
import { EMA } from './ema';

export class ZLEMA extends MovingAverageBase {
    calculate(): MovingAverageOutput {
        const lag = Math.floor((this.period - 1) / 2);
        const src = this.prices.map((price, index) => (index >= lag ? 2 * price - this.prices[index - lag] : price));
        const result = new EMA({ prices: src, period: this.period }).calculate().result;
        return { result };
    }
}
