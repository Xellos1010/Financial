import { MovingAverageBase } from './movingAverageBase';
import { MovingAverageOutput } from '../types/movingAverages';
import { EMA } from './ema';

export class TEMA extends MovingAverageBase {
    calculate(): MovingAverageOutput {
        const ema1 = new EMA({ prices: this.prices, period: this.period }).calculate().result;
        const ema2 = new EMA({ prices: ema1, period: this.period }).calculate().result;
        const ema3 = new EMA({ prices: ema2, period: this.period }).calculate().result;
        const result = ema1.map((value, index) => 3 * (value - ema2[index]) + ema3[index]);
        return { result };
    }
}
