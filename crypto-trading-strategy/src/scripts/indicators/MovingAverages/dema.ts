import { MovingAverageBase } from './movingAverageBase';
import { MovingAverageOutput } from '../types/movingAverages';
import { EMA } from './ema';

export class DEMA extends MovingAverageBase {
    calculate(): MovingAverageOutput {
        const ema1 = new EMA({ prices: this.prices, period: this.period }).calculate().result;
        const ema2 = new EMA({ prices: ema1, period: this.period }).calculate().result;
        const result = ema1.map((value, index) => 2 * value - ema2[index]);
        return { result };
    }
}
